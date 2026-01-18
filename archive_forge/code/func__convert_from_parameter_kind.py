import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def _convert_from_parameter_kind(kind):
    if kind == Parameter.POSITIONAL_ONLY:
        return 0
    if kind == Parameter.POSITIONAL_OR_KEYWORD:
        return 1
    if kind == Parameter.VAR_POSITIONAL:
        return 2
    if kind == Parameter.KEYWORD_ONLY:
        return 3
    if kind == Parameter.VAR_KEYWORD:
        return 4