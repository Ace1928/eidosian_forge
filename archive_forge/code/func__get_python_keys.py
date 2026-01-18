import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def _get_python_keys(dicts):
    for dct in dicts:
        if dct.array_type == 'dict':
            for key in dct.get_key_values():
                dict_key = key.get_safe_value(default=_sentinel)
                if dict_key is not _sentinel:
                    yield dict_key