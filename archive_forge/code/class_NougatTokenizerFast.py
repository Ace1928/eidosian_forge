import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class NougatTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. This class mainly adds Nougat-specific
    methods for postprocessing the generated text.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.

        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = None

    def __init__(self, vocab_file=None, tokenizer_file=None, clean_up_tokenization_spaces=False, unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', **kwargs):
        super().__init__(vocab_file=vocab_file, tokenizer_file=tokenizer_file, clean_up_tokenization_spaces=clean_up_tokenization_spaces, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)
        self.vocab_file = vocab_file

    def remove_hallucinated_references(self, text: str) -> str:
        """
        Remove hallucinated or missing references from the text.

        This function identifies and removes references that are marked as missing or hallucinated from the input text.

        Args:
            text (`str`):
                The input text containing references.

        Returns:
            `str`: The text with hallucinated references removed.
        """
        lines = text.split('\n')
        if len(lines) == 0:
            return ''
        clean_lines = remove_numbers(lines)
        slices = get_slices(lines, clean_lines)
        to_delete = []
        for slice in slices:
            to_delete.append(remove_slice_from_lines(lines, clean_lines, slice))
        for to_delete in reversed(to_delete):
            text = text.replace(to_delete, '\n\n[MISSING_PAGE_POST]\n\n')
        text = re.sub('## References\\n+\\[MISSING_PAGE_POST(:\\d+)?\\]', '\n\n[MISSING_PAGE_POST\\1]', text)
        return text

    def correct_tables(self, generation: str) -> str:
        """
        Takes a generated string and fixes tables/tabulars to make them match the markdown format needed.

        Args:
            generation (str): The generated text to be postprocessed.

        Returns:
            str: The postprocessed text.

        Example:

        ```python
        correct_tables("\\begin{table} \\begin{tabular}{l l} & \\ \\end{tabular} \\end{table}")
        "\\begin{table}
\\begin{tabular}{l l} & \\ \\end{tabular}
\\end{table}"
        ```
        """
        for l in generation.split('\n'):
            if l.count('\\begin{tabular}') > 15 or l.count('\\multicolumn') > 60 or l.count('&') > 400:
                generation = generation.replace(l, '')
        generation = generation.replace('\\begin{table} \\begin{tabular}', '\\begin{table}\n\\begin{tabular}')
        generation = generation.replace('\\end{tabular} \\end{table}', '\\end{tabular}\n\\end{table}')
        generation = generation.replace('\\end{table} Tab', '\\end{table}\nTab')
        generation = re.sub('(^.+)\\\\begin{tab', '\\1\\n\\\\begin{tab', generation, flags=re.M)
        generation = generation.replace('\\begin{tabular}{l l}  & \\\\ \\end{tabular}', '')
        generation = generation.replace('\\begin{tabular}{}\n\n\\end{tabular}', '')
        return generation

    def post_process_single(self, generation: str, fix_markdown: bool=True) -> str:
        """
        Postprocess a single generated text. Regular expressions used here are taken directly from the Nougat article
        authors. These expressions are commented for clarity and tested end-to-end in most cases.

        Args:
            generation (str): The generated text to be postprocessed.
            fix_markdown (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

        Returns:
            str: The postprocessed text.
        """
        generation = re.sub('(?:\\n|^)#+ \\d*\\W? ?(.{100,})', '\\n\\1', generation)
        generation = generation.strip()
        generation = generation.replace('\n* [leftmargin=*]\n', '\n')
        generation = re.sub('^#+ (?:\\.?(?:\\d|[ixv])+)*\\s*(?:$|\\n\\s*)', '', generation, flags=re.M)
        lines = generation.split('\n')
        if lines[-1].startswith('#') and lines[-1].lstrip('#').startswith(' ') and (len(lines) > 1):
            logger.info('Likely hallucinated title at the end of the page: ' + lines[-1])
            generation = '\n'.join(lines[:-1])
        generation = truncate_repetitions(generation)
        generation = self.remove_hallucinated_references(generation)
        generation = re.sub('^\\* \\[\\d+\\](\\s?[A-W]\\.+\\s?){10,}.*$', '', generation, flags=re.M)
        generation = re.sub('^(\\* \\[\\d+\\])\\[\\](.*)$', '\\1\\2', generation, flags=re.M)
        generation = re.sub('(^\\w\\n\\n|\\n\\n\\w$)', '', generation)
        generation = re.sub('([\\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\\s.,:()])', '\\1\\(\\2_{\\3}\\)\\4', generation)
        generation = re.sub('([\\s.,\\d])_([a-zA-Z0-9])_([\\s.,\\d;])', '\\1\\(\\2\\)\\3', generation)
        generation = re.sub('(\\nFootnote .*?:) (?:footnotetext|thanks):\\W*(.*(?:\\n\\n|$))', '\\1 \\2', generation)
        generation = re.sub('\\[FOOTNOTE:.+?\\](.*?)\\[ENDFOOTNOTE\\]', '', generation)
        generation = normalize_list_like_lines(generation)
        if generation.endswith(('.', '}')):
            generation += '\n\n'
        if re.match('[A-Z0-9,;:]$', generation):
            generation += ' '
        elif generation.startswith(('#', '**', '\\begin')):
            generation = '\n\n' + generation
        elif generation.split('\n')[-1].startswith(('#', 'Figure', 'Table')):
            generation = generation + '\n\n'
        else:
            try:
                last_word = generation.split(' ')[-1]
                if last_word in nltk.corpus.words.words():
                    generation += ' '
            except LookupError:
                generation += ' '
        generation = self.correct_tables(generation)
        generation = generation.replace('\\begin{array}[]{', '\\begin{array}{')
        generation = re.sub('\\\\begin{tabular}{([clr ]){2,}}\\s*[& ]*\\s*(\\\\\\\\)? \\\\end{tabular}', '', generation)
        generation = re.sub('(\\*\\*S\\. A\\. B\\.\\*\\*\\n+){2,}', '', generation)
        generation = re.sub('^#+( [\\[\\d\\w])?$', '', generation, flags=re.M)
        generation = re.sub('^\\.\\s*$', '', generation, flags=re.M)
        generation = re.sub('\\n{3,}', '\n\n', generation)
        if fix_markdown:
            return markdown_compatible(generation)
        else:
            return generation

    def post_process_generation(self, generation: Union[str, List[str]], fix_markdown: bool=True, num_workers: int=None) -> Union[str, List[str]]:
        """
        Postprocess a generated text or a list of generated texts.

        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

        Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.

        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional*):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).

        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
        requires_backends(self, ['nltk', 'levenshtein'])
        if isinstance(generation, list):
            if num_workers is not None and isinstance(num_workers, int):
                with Pool(num_workers) as p:
                    return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
            else:
                return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
        else:
            return self.post_process_single(generation, fix_markdown=fix_markdown)