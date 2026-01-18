import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def _get_string_prefix_and_quote(string):
    match = re.match('(\\w*)("""|\\\'{3}|"|\\\')', string)
    if match is None:
        return (None, None)
    return (match.group(1), match.group(2))