import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def _create_repr_string(literal_string, dict_key):
    if not isinstance(dict_key, (str, bytes)) or not literal_string:
        return repr(dict_key)
    r = repr(dict_key)
    prefix, quote = _get_string_prefix_and_quote(literal_string)
    if quote is None:
        return r
    if quote == r[0]:
        return prefix + r
    return prefix + quote + r[1:-1] + quote