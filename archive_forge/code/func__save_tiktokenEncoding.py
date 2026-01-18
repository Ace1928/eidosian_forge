import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def _save_tiktokenEncoding(pickler, obj):
    import tiktoken
    log(pickler, f'Enc: {obj}')
    args = (obj.name, obj._pat_str, obj._mergeable_ranks, obj._special_tokens)
    pickler.save_reduce(tiktoken.Encoding, args, obj=obj)
    log(pickler, '# Enc')