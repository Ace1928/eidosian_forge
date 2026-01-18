from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
def _remove_given_params(arguments, param_names):
    count = 0
    used_keys = set()
    for key, _ in arguments.unpack():
        if key is None:
            count += 1
        else:
            used_keys.add(key)
    for p in param_names:
        if count and p.maybe_positional_argument():
            count -= 1
            continue
        if p.string_name in used_keys and p.maybe_keyword_argument():
            continue
        yield p