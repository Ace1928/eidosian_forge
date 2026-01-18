import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
@classmethod
def _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, token=None, cache_dir=None, local_files_only=False, _commit_hash=None, _is_local=False, **kwargs):
    tokenizer = super()._from_pretrained(resolved_vocab_files, pretrained_model_name_or_path, init_configuration, *init_inputs, token=token, cache_dir=cache_dir, local_files_only=local_files_only, _commit_hash=_commit_hash, _is_local=_is_local, **kwargs)
    tokenizer.set_src_lang_special_tokens(tokenizer._src_lang)
    tokenizer.set_tgt_lang_special_tokens(tokenizer._tgt_lang)
    return tokenizer