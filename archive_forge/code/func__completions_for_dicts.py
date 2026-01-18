import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def _completions_for_dicts(inference_state, dicts, literal_string, cut_end_quote, fuzzy):
    for dict_key in sorted(_get_python_keys(dicts), key=lambda x: repr(x)):
        dict_key_str = _create_repr_string(literal_string, dict_key)
        if dict_key_str.startswith(literal_string):
            name = StringName(inference_state, dict_key_str[:-len(cut_end_quote) or None])
            yield Completion(inference_state, name, stack=None, like_name_length=len(literal_string), is_fuzzy=fuzzy)