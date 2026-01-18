import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
class TapasTokenizer(PreTrainedTokenizer):
    """
    Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by
    TAPAS models.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods. [`TapasTokenizer`] creates several token type ids to
    encode tabular structure. To be more precise, it adds 7 token type ids, in the following order: `segment_ids`,
    `column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks` and `numeric_relations`:

    - segment_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and
      padding.
    - column_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
    - row_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    - prev_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in
      a conversational setup (such as SQA).
    - column_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a
      column "number of movies" with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2
      respectively. 0 for all question tokens, special tokens and padding.
    - inv_column_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if
      you have a column "number of movies" with values 87, 53 and 69, then the inverse column ranks of these tokens are
      1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
    - numeric_relations: indicate numeric relations between the question and the tokens of the table. 0 for all
      question tokens, special tokens and padding.

    [`TapasTokenizer`] runs end-to-end tokenization on a table and associated sentences: punctuation splitting and
    wordpiece.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        empty_token (`str`, *optional*, defaults to `"[EMPTY]"`):
            The token used for empty cell values in a table. Empty cell values include "", "n/a", "nan" and "?".
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        cell_trim_length (`int`, *optional*, defaults to -1):
            If > 0: Trim cells so that the length is <= this value. Also disables further cell trimming, should thus be
            used with `truncation` set to `True`.
        max_column_id (`int`, *optional*):
            Max column id to extract.
        max_row_id (`int`, *optional*):
            Max row id to extract.
        strip_column_names (`bool`, *optional*, defaults to `False`):
            Whether to add empty strings instead of column names.
        update_answer_coordinates (`bool`, *optional*, defaults to `False`):
            Whether to recompute the answer coordinates from the answer text.
        min_question_length (`int`, *optional*):
            Minimum length of each question in terms of tokens (will be skipped otherwise).
        max_question_length (`int`, *optional*):
            Maximum length of each question in terms of tokens (will be skipped otherwise).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', empty_token='[EMPTY]', tokenize_chinese_chars=True, strip_accents=None, cell_trim_length: int=-1, max_column_id: int=None, max_row_id: int=None, strip_column_names: bool=False, update_answer_coordinates: bool=False, min_question_length=None, max_question_length=None, model_max_length: int=512, additional_special_tokens: Optional[List[str]]=None, **kwargs):
        if not is_pandas_available():
            raise ImportError('Pandas is required for the TAPAS tokenizer.')
        if additional_special_tokens is not None:
            if empty_token not in additional_special_tokens:
                additional_special_tokens.append(empty_token)
        else:
            additional_special_tokens = [empty_token]
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        self.cell_trim_length = cell_trim_length
        self.max_column_id = max_column_id if max_column_id is not None else model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.max_row_id = max_row_id if max_row_id is not None else model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.strip_column_names = strip_column_names
        self.update_answer_coordinates = update_answer_coordinates
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        super().__init__(do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize, never_split=never_split, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, empty_token=empty_token, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, cell_trim_length=cell_trim_length, max_column_id=max_column_id, max_row_id=max_row_id, strip_column_names=strip_column_names, update_answer_coordinates=update_answer_coordinates, min_question_length=min_question_length, max_question_length=max_question_length, model_max_length=model_max_length, additional_special_tokens=additional_special_tokens, **kwargs)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        if format_text(text) == EMPTY_TEXT:
            return [self.additional_special_tokens[0]]
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        else:
            vocab_file = (filename_prefix + '-' if filename_prefix else '') + save_directory
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!')
                    index = token_index
                writer.write(token + '\n')
                index += 1
        return (vocab_file,)

    def create_attention_mask_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the attention mask according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the attention mask values.
        """
        return [1] * (1 + len(query_ids) + 1 + len(table_values))

    def create_segment_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the segment token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the segment token type IDs values.
        """
        table_ids = list(zip(*table_values))[0] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + [1] * len(table_ids)

    def create_column_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the column token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the column token type IDs values.
        """
        table_column_ids = list(zip(*table_values))[1] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + list(table_column_ids)

    def create_row_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the row token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the row token type IDs values.
        """
        table_row_ids = list(zip(*table_values))[2] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Build model inputs from a question and flattened table for question answering or sequence classification tasks
        by concatenating and adding special tokens.

        Args:
            token_ids_0 (`List[int]`): The ids of the question.
            token_ids_1 (`List[int]`, *optional*): The ids of the flattened table.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            raise ValueError('With TAPAS, you must provide both question IDs and table IDs.')
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of question IDs.
            token_ids_1 (`List[int]`, *optional*):
                List of flattened table IDs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1)
        return [1] + [0] * len(token_ids_0) + [1]

    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(self, table: 'pd.DataFrame', queries: Optional[Union[TextInput, PreTokenizedInput, EncodedInput, List[TextInput], List[PreTokenizedInput], List[EncodedInput]]]=None, answer_coordinates: Optional[Union[List[Tuple], List[List[Tuple]]]]=None, answer_text: Optional[Union[List[TextInput], List[List[TextInput]]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) related to a table.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            queries (`str` or `List[str]`):
                Question or batch of questions related to a table to be encoded. Note that in case of a batch, all
                questions must refer to the **same** table.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. In case only a single table-question pair
                is provided, then the answer_coordinates must be a single list of one or more tuples. Each tuple must
                be a (row_index, column_index) pair. The first data row (not the column header row) has index 0. The
                first column has index 0. In case a batch of table-question pairs is provided, then the
                answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. In case only a single table-question pair is
                provided, then the answer_text must be a single list of one or more strings. Each string must be the
                answer text of a corresponding answer coordinate. In case a batch of table-question pairs is provided,
                then the answer_coordinates must be a list of lists of strings (each list corresponding to a single
                table-question pair).
        """
        assert isinstance(table, pd.DataFrame), 'Table must be of type pd.DataFrame'
        valid_query = False
        if queries is None or isinstance(queries, str):
            valid_query = True
        elif isinstance(queries, (list, tuple)):
            if len(queries) == 0 or isinstance(queries[0], str):
                valid_query = True
        if not valid_query:
            raise ValueError('queries input must of type `str` (single example), `List[str]` (batch or single pretokenized example). ')
        is_batched = isinstance(queries, (list, tuple))
        if is_batched:
            return self.batch_encode_plus(table=table, queries=queries, answer_coordinates=answer_coordinates, answer_text=answer_text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        else:
            return self.encode_plus(table=table, query=queries, answer_coordinates=answer_coordinates, answer_text=answer_text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(self, table: 'pd.DataFrame', queries: Optional[Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]]]=None, answer_coordinates: Optional[List[List[Tuple]]]=None, answer_text: Optional[List[List[TextInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        """
        Prepare a table and a list of strings for the model.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            queries (`List[str]`):
                Batch of questions related to a table to be encoded. Note that all questions must refer to the **same**
                table.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. Each tuple must be a (row_index,
                column_index) pair. The first data row (not the column header row) has index 0. The first column has
                index 0. The answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. In case a batch of table-question pairs is
                provided, then the answer_coordinates must be a list of lists of strings (each list corresponding to a
                single table-question pair). Each string must be the answer text of a corresponding answer coordinate.
        """
        if return_token_type_ids is not None and (not add_special_tokens):
            raise ValueError('Asking to return token_type_ids while setting add_special_tokens to False results in an undefined behavior. Please set add_special_tokens to True or set return_token_type_ids to None.')
        if answer_coordinates and (not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError('In case you provide answers, both answer_coordinates and answer_text should be provided')
        elif answer_coordinates is None and answer_text is None:
            answer_coordinates = answer_text = [None] * len(queries)
        if 'is_split_into_words' in kwargs:
            raise NotImplementedError('Currently TapasTokenizer only supports questions as strings.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.')
        return self._batch_encode_plus(table=table, queries=queries, answer_coordinates=answer_coordinates, answer_text=answer_text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _get_question_tokens(self, query):
        """Tokenizes the query, taking into account the max and min question length."""
        query_tokens = self.tokenize(query)
        if self.max_question_length is not None and len(query_tokens) > self.max_question_length:
            logger.warning('Skipping query as its tokens are longer than the max question length')
            return ('', [])
        if self.min_question_length is not None and len(query_tokens) < self.min_question_length:
            logger.warning('Skipping query as its tokens are shorter than the min question length')
            return ('', [])
        return (query, query_tokens)

    def _batch_encode_plus(self, table, queries: Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]], answer_coordinates: Optional[List[List[Tuple]]]=None, answer_text: Optional[List[List[TextInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=True, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        table_tokens = self._tokenize_table(table)
        queries_tokens = []
        for idx, query in enumerate(queries):
            query, query_tokens = self._get_question_tokens(query)
            queries[idx] = query
            queries_tokens.append(query_tokens)
        batch_outputs = self._batch_prepare_for_model(table, queries, tokenized_table=table_tokens, queries_tokens=queries_tokens, answer_coordinates=answer_coordinates, padding=padding, truncation=truncation, answer_text=answer_text, add_special_tokens=add_special_tokens, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, prepend_batch_axis=True, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, verbose=verbose)
        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(self, raw_table: 'pd.DataFrame', raw_queries: Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]], tokenized_table: Optional[TokenizedTable]=None, queries_tokens: Optional[List[List[str]]]=None, answer_coordinates: Optional[List[List[Tuple]]]=None, answer_text: Optional[List[List[TextInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=True, return_attention_mask: Optional[bool]=True, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, prepend_batch_axis: bool=False, **kwargs) -> BatchEncoding:
        batch_outputs = {}
        for index, example in enumerate(zip(raw_queries, queries_tokens, answer_coordinates, answer_text)):
            raw_query, query_tokens, answer_coords, answer_txt = example
            outputs = self.prepare_for_model(raw_table, raw_query, tokenized_table=tokenized_table, query_tokens=query_tokens, answer_coordinates=answer_coords, answer_text=answer_txt, add_special_tokens=add_special_tokens, padding=PaddingStrategy.DO_NOT_PAD.value, truncation=truncation, max_length=max_length, pad_to_multiple_of=None, return_attention_mask=False, return_token_type_ids=return_token_type_ids, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=None, prepend_batch_axis=False, verbose=verbose, prev_answer_coordinates=answer_coordinates[index - 1] if index != 0 else None, prev_answer_text=answer_text[index - 1] if index != 0 else None)
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = self.pad(batch_outputs, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(self, table: 'pd.DataFrame', query: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> List[int]:
        """
        Prepare a table and a string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use that method if you want to build your processing on
        your own, otherwise refer to `__call__`.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
        """
        encoded_inputs = self.encode_plus(table, query=query, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors, **kwargs)
        return encoded_inputs['input_ids']

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(self, table: 'pd.DataFrame', query: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, answer_coordinates: Optional[List[Tuple]]=None, answer_text: Optional[List[TextInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        """
        Prepare a table and a string for the model.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        if return_token_type_ids is not None and (not add_special_tokens):
            raise ValueError('Asking to return token_type_ids while setting add_special_tokens to False results in an undefined behavior. Please set add_special_tokens to True or set return_token_type_ids to None.')
        if answer_coordinates and (not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError('In case you provide answers, both answer_coordinates and answer_text should be provided')
        if 'is_split_into_words' in kwargs:
            raise NotImplementedError('Currently TapasTokenizer only supports questions as strings.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.')
        return self._encode_plus(table=table, query=query, answer_coordinates=answer_coordinates, answer_text=answer_text, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _encode_plus(self, table: 'pd.DataFrame', query: Union[TextInput, PreTokenizedInput, EncodedInput], answer_coordinates: Optional[List[Tuple]]=None, answer_text: Optional[List[TextInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=True, return_attention_mask: Optional[bool]=True, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs):
        if query is None:
            query = ''
            logger.warning('TAPAS is a question answering model but you have not passed a query. Please be aware that the model will probably not behave correctly.')
        table_tokens = self._tokenize_table(table)
        query, query_tokens = self._get_question_tokens(query)
        return self.prepare_for_model(table, query, tokenized_table=table_tokens, query_tokens=query_tokens, answer_coordinates=answer_coordinates, answer_text=answer_text, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, prepend_batch_axis=True, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, verbose=verbose)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(self, raw_table: 'pd.DataFrame', raw_query: Union[TextInput, PreTokenizedInput, EncodedInput], tokenized_table: Optional[TokenizedTable]=None, query_tokens: Optional[TokenizedTable]=None, answer_coordinates: Optional[List[Tuple]]=None, answer_text: Optional[List[TextInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TapasTruncationStrategy]=False, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=True, return_attention_mask: Optional[bool]=True, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, prepend_batch_axis: bool=False, **kwargs) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens.

        Args:
            raw_table (`pd.DataFrame`):
                The original table before any transformation (like tokenization) was applied to it.
            raw_query (`TextInput` or `PreTokenizedInput` or `EncodedInput`):
                The original query before any transformation (like tokenization) was applied to it.
            tokenized_table (`TokenizedTable`):
                The table after tokenization.
            query_tokens (`List[str]`):
                The query after tokenization.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        if isinstance(padding, bool):
            if padding and (max_length is not None or pad_to_multiple_of is not None):
                padding = PaddingStrategy.MAX_LENGTH
            else:
                padding = PaddingStrategy.DO_NOT_PAD
        elif not isinstance(padding, PaddingStrategy):
            padding = PaddingStrategy(padding)
        if isinstance(truncation, bool):
            if truncation:
                truncation = TapasTruncationStrategy.DROP_ROWS_TO_FIT
            else:
                truncation = TapasTruncationStrategy.DO_NOT_TRUNCATE
        elif not isinstance(truncation, TapasTruncationStrategy):
            truncation = TapasTruncationStrategy(truncation)
        encoded_inputs = {}
        is_part_of_batch = False
        prev_answer_coordinates, prev_answer_text = (None, None)
        if 'prev_answer_coordinates' in kwargs and 'prev_answer_text' in kwargs:
            is_part_of_batch = True
            prev_answer_coordinates = kwargs['prev_answer_coordinates']
            prev_answer_text = kwargs['prev_answer_text']
        num_rows = self._get_num_rows(raw_table, truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE)
        num_columns = self._get_num_columns(raw_table)
        _, _, num_tokens = self._get_table_boundaries(tokenized_table)
        if truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            num_rows, num_tokens = self._get_truncated_table_rows(query_tokens, tokenized_table, num_rows, num_columns, max_length, truncation_strategy=truncation)
        table_data = list(self._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))
        query_ids = self.convert_tokens_to_ids(query_tokens)
        table_ids = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))
        table_ids = self.convert_tokens_to_ids(list(table_ids))
        if 'return_overflowing_tokens' in kwargs and kwargs['return_overflowing_tokens']:
            raise ValueError('TAPAS does not return overflowing tokens as it works on tables.')
        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(query_ids, table_ids)
        else:
            input_ids = query_ids + table_ids
        if max_length is not None and len(input_ids) > max_length:
            raise ValueError(f'Could not encode the query and table header given the maximum length. Encoding the query and table header results in a length of {len(input_ids)} which is higher than the max_length of {max_length}')
        encoded_inputs['input_ids'] = input_ids
        segment_ids = self.create_segment_token_type_ids_from_sequences(query_ids, table_data)
        column_ids = self.create_column_token_type_ids_from_sequences(query_ids, table_data)
        row_ids = self.create_row_token_type_ids_from_sequences(query_ids, table_data)
        if not is_part_of_batch or (prev_answer_coordinates is None and prev_answer_text is None):
            prev_labels = [0] * len(row_ids)
        else:
            prev_labels = self.get_answer_ids(column_ids, row_ids, table_data, prev_answer_text, prev_answer_coordinates)
        raw_table = add_numeric_table_values(raw_table)
        raw_query = add_numeric_values_to_question(raw_query)
        column_ranks, inv_column_ranks = self._get_numeric_column_ranks(column_ids, row_ids, raw_table)
        numeric_relations = self._get_numeric_relations(raw_query, column_ids, row_ids, raw_table)
        if return_token_type_ids is None:
            return_token_type_ids = 'token_type_ids' in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        if return_attention_mask:
            attention_mask = self.create_attention_mask_from_sequences(query_ids, table_data)
            encoded_inputs['attention_mask'] = attention_mask
        if answer_coordinates is not None and answer_text is not None:
            labels = self.get_answer_ids(column_ids, row_ids, table_data, answer_text, answer_coordinates)
            numeric_values = self._get_numeric_values(raw_table, column_ids, row_ids)
            numeric_values_scale = self._get_numeric_values_scale(raw_table, column_ids, row_ids)
            encoded_inputs['labels'] = labels
            encoded_inputs['numeric_values'] = numeric_values
            encoded_inputs['numeric_values_scale'] = numeric_values_scale
        if return_token_type_ids:
            token_type_ids = [segment_ids, column_ids, row_ids, prev_labels, column_ranks, inv_column_ranks, numeric_relations]
            token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
            encoded_inputs['token_type_ids'] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(query_ids, table_ids)
            else:
                encoded_inputs['special_tokens_mask'] = [0] * len(input_ids)
        if max_length is None and len(encoded_inputs['input_ids']) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get('sequence-length-is-longer-than-the-specified-maximum', False):
                logger.warning(f'Token indices sequence length is longer than the specified maximum sequence length for this model ({len(encoded_inputs['input_ids'])} > {self.model_max_length}). Running this sequence through the model will result in indexing errors.')
            self.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True
        if padding != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(encoded_inputs, max_length=max_length, padding=padding.value, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        if return_length:
            encoded_inputs['length'] = len(encoded_inputs['input_ids'])
        batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)
        return batch_outputs

    def _get_truncated_table_rows(self, query_tokens: List[str], tokenized_table: TokenizedTable, num_rows: int, num_columns: int, max_length: int, truncation_strategy: Union[str, TapasTruncationStrategy]) -> Tuple[int, int]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            query_tokens (`List[str]`):
                List of strings corresponding to the tokenized query.
            tokenized_table (`TokenizedTable`):
                Tokenized table
            num_rows (`int`):
                Total number of table rows
            num_columns (`int`):
                Total number of table columns
            max_length (`int`):
                Total maximum length.
            truncation_strategy (`str` or [`TapasTruncationStrategy`]):
                Truncation strategy to use. Seeing as this method should only be called when truncating, the only
                available strategy is the `"drop_rows_to_fit"` strategy.

        Returns:
            `Tuple(int, int)`: tuple containing the number of rows after truncation, and the number of tokens available
            for each table element.
        """
        if not isinstance(truncation_strategy, TapasTruncationStrategy):
            truncation_strategy = TapasTruncationStrategy(truncation_strategy)
        if max_length is None:
            max_length = self.model_max_length
        if truncation_strategy == TapasTruncationStrategy.DROP_ROWS_TO_FIT:
            while True:
                num_tokens = self._get_max_num_tokens(query_tokens, tokenized_table, num_rows=num_rows, num_columns=num_columns, max_length=max_length)
                if num_tokens is not None:
                    break
                num_rows -= 1
                if num_rows < 1:
                    break
        elif truncation_strategy != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            raise ValueError(f'Unknown truncation strategy {truncation_strategy}.')
        return (num_rows, num_tokens or 1)

    def _tokenize_table(self, table=None):
        """
        Tokenizes column headers and cell texts of a table.

        Args:
            table (`pd.Dataframe`):
                Table. Returns: `TokenizedTable`: TokenizedTable object.
        """
        tokenized_rows = []
        tokenized_row = []
        for column in table:
            if self.strip_column_names:
                tokenized_row.append(self.tokenize(''))
            else:
                tokenized_row.append(self.tokenize(column))
        tokenized_rows.append(tokenized_row)
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                tokenized_row.append(self.tokenize(cell))
            tokenized_rows.append(tokenized_row)
        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(TokenCoordinates(row_index=row_index, column_index=column_index, token_index=token_index))
        return TokenizedTable(rows=tokenized_rows, selected_tokens=token_coordinates)

    def _question_encoding_cost(self, question_tokens):
        return len(question_tokens) + 2

    def _get_token_budget(self, question_tokens, max_length=None):
        """
        Computes the number of tokens left for the table after tokenizing a question, taking into account the max
        sequence length of the model.

        Args:
            question_tokens (`List[String]`):
                List of question tokens. Returns: `int`: the number of tokens left for the table, given the model max
                length.
        """
        return (max_length if max_length is not None else self.model_max_length) - self._question_encoding_cost(question_tokens)

    def _get_table_values(self, table, num_columns, num_rows, num_tokens) -> Generator[TableValue, None, None]:
        """Iterates over partial table and returns token, column and row indexes."""
        for tc in table.selected_tokens:
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            while word_begin_index >= 0 and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield TableValue(token, tc.column_index + 1, tc.row_index)

    def _get_table_boundaries(self, table):
        """Return maximal number of rows, columns and tokens."""
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        return (max_num_rows, max_num_columns, max_num_tokens)

    def _get_table_cost(self, table, num_columns, num_rows, num_tokens):
        return sum((1 for _ in self._get_table_values(table, num_columns, num_rows, num_tokens)))

    def _get_max_num_tokens(self, question_tokens, tokenized_table, num_columns, num_rows, max_length):
        """Computes max number of tokens that can be squeezed into the budget."""
        token_budget = self._get_token_budget(question_tokens, max_length)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows, num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self.cell_trim_length >= 0:
                return None
            if num_tokens == 0:
                return None
        return num_tokens

    def _get_num_columns(self, table):
        num_columns = table.shape[1]
        if num_columns >= self.max_column_id:
            raise ValueError('Too many columns')
        return num_columns

    def _get_num_rows(self, table, drop_rows_to_fit):
        num_rows = table.shape[0]
        if num_rows >= self.max_row_id:
            if drop_rows_to_fit:
                num_rows = self.max_row_id - 1
            else:
                raise ValueError('Too many rows')
        return num_rows

    def _serialize_text(self, question_tokens):
        """Serializes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)
        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)
        return (tokens, segment_ids, column_ids, row_ids)

    def _serialize(self, question_tokens, table, num_columns, num_rows, num_tokens):
        """Serializes table and text."""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(question_tokens)
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)
        for token, column_id, row_id in self._get_table_values(table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)
        return SerializedExample(tokens=tokens, segment_ids=segment_ids, column_ids=column_ids, row_ids=row_ids)

    def _get_column_values(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    def _get_cell_token_indexes(self, column_ids, row_ids, column_id, row_id):
        for index in range(len(column_ids)):
            if column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id:
                yield index

    def _get_numeric_column_ranks(self, column_ids, row_ids, table):
        """Returns column ranks for all numeric columns."""
        ranks = [0] * len(column_ids)
        inv_ranks = [0] * len(column_ids)
        if table is not None:
            for col_index in range(len(table.columns)):
                table_numeric_values = self._get_column_values(table, col_index)
                if not table_numeric_values:
                    continue
                try:
                    key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
                except ValueError:
                    continue
                table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}
                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)
                unique_values = sorted(table_numeric_values_inv.keys())
                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank
        return (ranks, inv_ranks)

    def _get_numeric_sort_key_fn(self, table_numeric_values, value):
        """
        Returns the sort key function for comparing value to table values. The function returned will be a suitable
        input for the key param of the sort(). See number_annotation_utils._get_numeric_sort_key_fn for details

        Args:
            table_numeric_values: Numeric values of a column
            value: Numeric value in the question

        Returns:
            A function key function to compare column and question values.
        """
        if not table_numeric_values:
            return None
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        try:
            return get_numeric_sort_key_fn(all_values)
        except ValueError:
            return None

    def _get_numeric_relations(self, question, column_ids, row_ids, table):
        """
        Returns numeric relations embeddings

        Args:
            question: Question object.
            column_ids: Maps word piece position to column id.
            row_ids: Maps word piece position to row id.
            table: The table containing the numeric cell values.
        """
        numeric_relations = [0] * len(column_ids)
        cell_indices_to_relations = collections.defaultdict(set)
        if question is not None and table is not None:
            for numeric_value_span in question.numeric_spans:
                for value in numeric_value_span.values:
                    for column_index in range(len(table.columns)):
                        table_numeric_values = self._get_column_values(table, column_index)
                        sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values, value)
                        if sort_key_fn is None:
                            continue
                        for row_index, cell_value in table_numeric_values.items():
                            relation = get_numeric_relation(value, cell_value, sort_key_fn)
                            if relation is not None:
                                cell_indices_to_relations[column_index, row_index].add(relation)
        for (column_index, row_index), relations in cell_indices_to_relations.items():
            relation_set_index = 0
            for relation in relations:
                assert relation.value >= Relation.EQ.value
                relation_set_index += 2 ** (relation.value - Relation.EQ.value)
            for cell_token_index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                numeric_relations[cell_token_index] = relation_set_index
        return numeric_relations

    def _get_numeric_values(self, table, column_ids, row_ids):
        """Returns numeric values for computation of answer loss."""
        numeric_values = [float('nan')] * len(column_ids)
        if table is not None:
            num_rows = table.shape[0]
            num_columns = table.shape[1]
            for col_index in range(num_columns):
                for row_index in range(num_rows):
                    numeric_value = table.iloc[row_index, col_index].numeric_value
                    if numeric_value is not None:
                        if numeric_value.float_value is None:
                            continue
                        float_value = numeric_value.float_value
                        if float_value == float('inf'):
                            continue
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            numeric_values[index] = float_value
        return numeric_values

    def _get_numeric_values_scale(self, table, column_ids, row_ids):
        """Returns a scale to each token to down weigh the value of long words."""
        numeric_values_scale = [1.0] * len(column_ids)
        if table is None:
            return numeric_values_scale
        num_rows = table.shape[0]
        num_columns = table.shape[1]
        for col_index in range(num_columns):
            for row_index in range(num_rows):
                indices = list(self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index))
                num_indices = len(indices)
                if num_indices > 1:
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)
        return numeric_values_scale

    def _pad_to_seq_length(self, inputs):
        while len(inputs) > self.model_max_length:
            inputs.pop()
        while len(inputs) < self.model_max_length:
            inputs.append(0)

    def _get_all_answer_ids_from_coordinates(self, column_ids, row_ids, answers_list):
        """Maps lists of answer coordinates to token indexes."""
        answer_ids = [0] * len(column_ids)
        found_answers = set()
        all_answers = set()
        for answers in answers_list:
            column_index, row_index = answers
            all_answers.add((column_index, row_index))
            for index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1
        missing_count = len(all_answers) - len(found_answers)
        return (answer_ids, missing_count)

    def _get_all_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """
        Maps answer coordinates of a question to token indexes.

        In the SQA format (TSV), the coordinates are given as (row, column) tuples. Here, we first swap them to
        (column, row) format before calling _get_all_answer_ids_from_coordinates.
        """

        def _to_coordinates(answer_coordinates_question):
            return [(coords[1], coords[0]) for coords in answer_coordinates_question]
        return self._get_all_answer_ids_from_coordinates(column_ids, row_ids, answers_list=_to_coordinates(answer_coordinates))

    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
        logging.info(f'text: {text} {segment}')
        for index in range(1 + len(text) - len(segment)):
            for seg_index, seg_token in enumerate(segment):
                if text[index + seg_index].piece != seg_token.piece:
                    break
            else:
                return index
        return None

    def _find_answer_coordinates_from_answer_text(self, tokenized_table, answer_text):
        """Returns all occurrences of answer_text in the table."""
        logging.info(f'answer text: {answer_text}')
        for row_index, row in enumerate(tokenized_table.rows):
            if row_index == 0:
                continue
            for col_index, cell in enumerate(row):
                token_index = self._find_tokens(cell, answer_text)
                if token_index is not None:
                    yield TokenCoordinates(row_index=row_index, column_index=col_index, token_index=token_index)

    def _find_answer_ids_from_answer_texts(self, column_ids, row_ids, tokenized_table, answer_texts):
        """Maps question with answer texts to the first matching token indexes."""
        answer_ids = [0] * len(column_ids)
        for answer_text in answer_texts:
            for coordinates in self._find_answer_coordinates_from_answer_text(tokenized_table, answer_text):
                indexes = list(self._get_cell_token_indexes(column_ids, row_ids, column_id=coordinates.column_index, row_id=coordinates.row_index - 1))
                indexes.sort()
                coordinate_answer_ids = []
                if indexes:
                    begin_index = coordinates.token_index + indexes[0]
                    end_index = begin_index + len(answer_text)
                    for index in indexes:
                        if index >= begin_index and index < end_index:
                            coordinate_answer_ids.append(index)
                if len(coordinate_answer_ids) == len(answer_text):
                    for index in coordinate_answer_ids:
                        answer_ids[index] = 1
                    break
        return answer_ids

    def _get_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """Maps answer coordinates of a question to token indexes."""
        answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids, answer_coordinates)
        if missing_count:
            raise ValueError("Couldn't find all answers")
        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(column_ids, row_ids, tokenized_table, answer_texts=[self.tokenize(at) for at in answer_texts_question])
        return self._get_answer_ids(column_ids, row_ids, answer_coordinates_question)

    def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs['input_ids'])
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_inputs['input_ids']) != max_length
        if return_attention_mask and 'attention_mask' not in encoded_inputs:
            encoded_inputs['attention_mask'] = [1] * len(encoded_inputs['input_ids'])
        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs['input_ids'])
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'] + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'] + [[self.pad_token_type_id] * 7] * difference
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = encoded_inputs['labels'] + [0] * difference
                if 'numeric_values' in encoded_inputs:
                    encoded_inputs['numeric_values'] = encoded_inputs['numeric_values'] + [float('nan')] * difference
                if 'numeric_values_scale' in encoded_inputs:
                    encoded_inputs['numeric_values_scale'] = encoded_inputs['numeric_values_scale'] + [1.0] * difference
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
                encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [0] * difference + encoded_inputs['attention_mask']
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [[self.pad_token_type_id] * 7] * difference + encoded_inputs['token_type_ids']
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = [0] * difference + encoded_inputs['labels']
                if 'numeric_values' in encoded_inputs:
                    encoded_inputs['numeric_values'] = [float('nan')] * difference + encoded_inputs['numeric_values']
                if 'numeric_values_scale' in encoded_inputs:
                    encoded_inputs['numeric_values_scale'] = [1.0] * difference + encoded_inputs['numeric_values_scale']
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [1] * difference + encoded_inputs['special_tokens_mask']
                encoded_inputs['input_ids'] = [self.pad_token_id] * difference + encoded_inputs['input_ids']
            else:
                raise ValueError('Invalid padding strategy:' + str(self.padding_side))
        return encoded_inputs

    def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            if col >= 0 and row >= 0 and (segment_id == 1):
                yield (i, p)

    def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
        """Computes average probability per cell, aggregating over tokens."""
        coords_to_probs = collections.defaultdict(list)
        for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            coords_to_probs[col, row].append(prob)
        return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}

    def convert_logits_to_predictions(self, data, logits, logits_agg=None, cell_classification_threshold=0.5):
        """
        Converts logits of [`TapasForQuestionAnswering`] to actual predicted answer coordinates and optional
        aggregation indices.

        The original implementation, on which this function is based, can be found
        [here](https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288).

        Args:
            data (`dict`):
                Dictionary mapping features to actual values. Should be created using [`TapasTokenizer`].
            logits (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor containing the logits at the token level.
            logits_agg (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, num_aggregation_labels)`, *optional*):
                Tensor containing the aggregation logits.
            cell_classification_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to be used for cell selection. All table cells for which their probability is larger than
                this threshold will be selected.

        Returns:
            `tuple` comprising various elements depending on the inputs:

            - predicted_answer_coordinates (`List[List[[tuple]]` of length `batch_size`): Predicted answer coordinates
              as a list of lists of tuples. Each element in the list contains the predicted answer coordinates of a
              single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index, column index).
            - predicted_aggregation_indices (`List[int]`of length `batch_size`, *optional*, returned when
              `logits_aggregation` is provided): Predicted aggregation operator indices of the aggregation head.
        """
        logits = logits.numpy()
        if logits_agg is not None:
            logits_agg = logits_agg.numpy()
        data = {key: value.numpy() for key, value in data.items() if key != 'training'}
        logits[logits < -88.7] = -88.7
        probabilities = 1 / (1 + np.exp(-logits)) * data['attention_mask']
        token_types = ['segment_ids', 'column_ids', 'row_ids', 'prev_labels', 'column_ranks', 'inv_column_ranks', 'numeric_relations']
        input_ids = data['input_ids']
        segment_ids = data['token_type_ids'][:, :, token_types.index('segment_ids')]
        row_ids = data['token_type_ids'][:, :, token_types.index('row_ids')]
        column_ids = data['token_type_ids'][:, :, token_types.index('column_ids')]
        num_batch = input_ids.shape[0]
        predicted_answer_coordinates = []
        for i in range(num_batch):
            probabilities_example = probabilities[i].tolist()
            segment_ids_example = segment_ids[i]
            row_ids_example = row_ids[i]
            column_ids_example = column_ids[i]
            max_width = column_ids_example.max()
            max_height = row_ids_example.max()
            if max_width == 0 and max_height == 0:
                continue
            cell_coords_to_prob = self._get_mean_cell_probs(probabilities_example, segment_ids_example.tolist(), row_ids_example.tolist(), column_ids_example.tolist())
            answer_coordinates = []
            for col in range(max_width):
                for row in range(max_height):
                    cell_prob = cell_coords_to_prob.get((col, row), None)
                    if cell_prob is not None:
                        if cell_prob > cell_classification_threshold:
                            answer_coordinates.append((row, col))
            answer_coordinates = sorted(answer_coordinates)
            predicted_answer_coordinates.append(answer_coordinates)
        output = (predicted_answer_coordinates,)
        if logits_agg is not None:
            predicted_aggregation_indices = logits_agg.argmax(axis=-1)
            output = (predicted_answer_coordinates, predicted_aggregation_indices.tolist())
        return output