import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
class TransfoXLTokenizer(PreTrainedTokenizer):
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        special (`List[str]`, *optional*):
            A list of special tokens (to be treated by the original implementation of this tokenizer).
        min_freq (`int`, *optional*, defaults to 0):
            The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it
            will be mapped to `unk_token`).
        max_size (`int`, *optional*):
            The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found
            after excluding the tokens according to the `min_freq` rule.
        lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        delimiter (`str`, *optional*):
            The delimiter used between tokens.
        vocab_file (`str`, *optional*):
            File containing the vocabulary (from the original implementation).
        pretrained_vocab_file (`str`, *optional*):
            File containing the vocabulary as saved with the `save_pretrained()` method.
        never_split (`List[str]`, *optional*):
            List of tokens that should never be split. If no list is specified, will simply use the existing special
            tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<formula>']`):
            A list of additional special tokens (for the HuggingFace functionality).
        language (`str`, *optional*, defaults to `"en"`):
            The language of this tokenizer (used for mose preprocessing).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids']

    def __init__(self, special=None, min_freq=0, max_size=None, lower_case=False, delimiter=None, vocab_file=None, pretrained_vocab_file: str=None, never_split=None, unk_token='<unk>', eos_token='<eos>', additional_special_tokens=['<formula>'], language='en', **kwargs):
        logger.error("`TransfoXL` was deprecated due to security issues linked to `pickle.load` in `TransfoXLTokenizer`. See more details on this model's documentation page: `https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/transfo-xl.md`.")
        requires_backends(self, 'sacremoses')
        if special is None:
            special = []
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.punctuation_symbols = '!"#$%&()*+,-./\\:;<=>?@[\\]^_`{|}~'
        self.punction_without_space_before_pattern = re.compile(f'[^\\s][{self.punctuation_symbols}]')
        self.punctuation_with_space_around_pattern = self._compile_space_around_punctuation_pattern()
        self.language = language
        self.moses_punct_normalizer = sm.MosesPunctNormalizer(language)
        self.moses_tokenizer = sm.MosesTokenizer(language)
        self.moses_detokenizer = sm.MosesDetokenizer(language)
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        try:
            vocab_dict = None
            if pretrained_vocab_file is not None:
                if not strtobool(os.environ.get('TRUST_REMOTE_CODE', 'False')):
                    raise ValueError("This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially malicious. It's recommended to never unpickle data that could have come from an untrusted source, or that could have been tampered with. If you already verified the pickle data and decided to use it, you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it.")
                with open(pretrained_vocab_file, 'rb') as f:
                    vocab_dict = pickle.load(f)
                if isinstance(vocab_dict, int):
                    if not is_torch_available():
                        raise ImportError('Not trying to load dict with PyTorch as you need to install pytorch to load from a PyTorch pretrained vocabulary, or activate it with environment variables USE_TORCH=1 and USE_TF=0.')
                    vocab_dict = torch.load(pretrained_vocab_file)
            if vocab_dict is not None:
                for key, value in vocab_dict.items():
                    if key not in self.__dict__ or key in ['sym2idx', 'idx2sym']:
                        self.__dict__[key] = value
            elif vocab_file is not None:
                self.build_vocab()
        except Exception as e:
            raise ValueError(f'Unable to parse file {pretrained_vocab_file}. Unknown format. If you tried to load a model saved through TransfoXLTokenizerFast, please note they are not compatible.') from e
        if vocab_file is not None:
            self.build_vocab()
        super().__init__(special=special, min_freq=min_freq, max_size=max_size, lower_case=lower_case, delimiter=delimiter, vocab_file=vocab_file, pretrained_vocab_file=pretrained_vocab_file, never_split=never_split, unk_token=unk_token, eos_token=eos_token, additional_special_tokens=additional_special_tokens, language=language, **kwargs)
        if never_split is None:
            never_split = self.all_special_tokens
        self.never_split = never_split

    @property
    def do_lower_case(self):
        return self.lower_case

    def _compile_space_around_punctuation_pattern(self):
        look_ahead_for_special_token = f'(?=[{self.punctuation_symbols}])'
        look_ahead_to_match_all_except_space = '(?=[^\\s])'
        return re.compile('' + look_ahead_for_special_token + look_ahead_to_match_all_except_space)

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose:
            logger.info(f'counting file {path} ...')
        assert os.path.exists(path), f'Input file {path} not found'
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and (idx % 500000 == 0):
                    logger.info(f'    line {idx}')
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)
        return sents

    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            logger.info(f'counting {len(sents)} sents ...')
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and (idx % 500000 == 0):
                logger.info(f'    line {idx}')
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        if '<UNK>' in self.sym2idx:
            self.unk_idx = self.sym2idx['<UNK>']
        elif '<unk>' in self.sym2idx:
            self.unk_idx = self.sym2idx['<unk>']
        else:
            raise ValueError('Token not in vocabulary and no <unk> token in vocabulary for replacement.')

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['pretrained_vocab_file'])
        else:
            vocab_file = (filename_prefix + '-' if filename_prefix else '') + save_directory
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.__dict__, f)
        return (vocab_file,)

    def build_vocab(self):
        if self.vocab_file:
            logger.info(f'building vocab from {self.vocab_file}')
            self._build_from_file(self.vocab_file)
            logger.info(f'Final vocab size {len(self.sym2idx)}')
        else:
            logger.info(f'building vocab with min_freq={self.min_freq}, max_size={self.max_size}')
            self.idx2sym = []
            self.sym2idx = OrderedDict()
            for sym in self.special:
                self.add_special(sym)
            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)
            logger.info(f'Final vocab size {len(self.sym2idx)} from {len(self.counter)} unique tokens')

    @torch_only_method
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
        if verbose:
            logger.info(f'encoding file {path} ...')
        assert os.path.exists(path), f'Output file {path} not found'
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and (idx % 500000 == 0):
                    logger.info(f'    line {idx}')
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))
        if ordered:
            encoded = torch.cat(encoded)
        return encoded

    @torch_only_method
    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose:
            logger.info(f'encoding {len(sents)} sents ...')
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and (idx % 500000 == 0):
                logger.info(f'    line {idx}')
            encoded.append(self.convert_to_tensor(symbols))
        if ordered:
            encoded = torch.cat(encoded)
        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, f'{sym.strip('<>')}_idx', self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def move_added_token(self, token: str, target_idx: int):
        """
        Moves an added token to a specific position in the vocab. This method should be used when resizing an embedding
        layer other than the last one in the `AdaptiveEmbedding` in order to move the token in the tokenizer from the
        default position (at the very end) to the desired one.

        Args:
            token: The token to move to a specific position in the vocab.
            target_idx: The position where the token should be moved to.
        """
        assert token in self.added_tokens_encoder, 'Token which should be moved has to be an added token'
        assert token not in self.idx2sym, 'Token which should be moved is already in vocab'
        self.idx2sym.insert(target_idx, token)
        self.sym2idx[token] = target_idx
        for idx in range(target_idx + 1, len(self.idx2sym)):
            current_sym = self.idx2sym[idx]
            self.sym2idx[current_sym] = idx
        old_index = self._added_tokens_encoder.pop(token)
        self._added_tokens_decoder.pop(old_index)

    def moses_punct_norm(self, text):
        return self.moses_punct_normalizer.normalize(text)

    def moses_tokenize(self, text):
        return self.moses_tokenizer.tokenize(text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split)

    def moses_pipeline(self, text: str) -> List[str]:
        """
        Does basic tokenization using [`sacremoses.MosesPunctNormalizer`] and [`sacremoses.MosesTokenizer`] with
        *aggressive_dash_splits=True* (see [`sacremoses.tokenize.MosesTokenizer.tokenize`]). Additionally, large
        comma-separated numbers and floating point values are split. E.g. "23,000 people are 1.80m tall" -> "23 @,@ 000
        people are 1 @.@ 80m tall"

        Args:
            text: Text to be tokenize

        Returns:
            A list of tokenized string

        Example:

        ```python
        >>> tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl/transfo-xl-wt103")
        >>> tokenizer.moses_pipeline("23,000 people are 1.80 m tall")
        ['23', '@,@', '000', 'people', 'are', '1', '@.@', '80', 'm', 'tall']
        ```"""
        text = self.moses_punct_norm(text)
        text = self.moses_tokenize(text)
        text = tokenize_numbers(text)
        return text

    def _convert_id_to_token(self, idx):
        """Converts an id in a token (BPE) using the vocab."""
        assert 0 <= idx < len(self), f'Index {idx} out of vocabulary range'
        return self.idx2sym[idx]

    def _convert_token_to_id(self, sym):
        """Converts a token (str) in an id using the vocab."""
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        elif hasattr(self, 'unk_idx'):
            return self.sym2idx.get(sym, self.unk_idx)
        elif '<unk>' in self.sym2idx:
            return self.sym2idx['<unk>']
        elif '<UNK>' in self.sym2idx:
            return self.sym2idx['<UNK>']
        else:
            raise ValueError('Token not in vocabulary and no <unk> token in vocabulary for replacement.')

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string. Additionally, the split numbers are converted back
        into it's original form.
        """
        out_string = self.moses_detokenizer.detokenize(tokens)
        return detokenize_numbers(out_string).strip()

    @torch_only_method
    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.convert_tokens_to_ids(symbols))

    @property
    def vocab_size(self):
        return len(self.idx2sym)

    def get_vocab(self):
        vocab = self.sym2idx.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        if self.lower_case:
            line = line.lower()
        if self.delimiter == '':
            symbols = line
        else:
            symbols = self.moses_pipeline(line)
        if add_double_eos:
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols