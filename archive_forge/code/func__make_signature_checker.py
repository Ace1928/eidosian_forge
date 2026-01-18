import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def _make_signature_checker(api_signature, signature):
    """Builds a PySignatureChecker for the given type signature.

  Args:
    api_signature: The `inspect.Signature` of the API whose signature is
      being checked.
    signature: Dictionary mapping parameter names to type annotations.

  Returns:
    A `PySignatureChecker`.
  """
    if not (isinstance(signature, dict) and all((isinstance(k, (str, int)) for k in signature))):
        raise TypeError('signatures must be dictionaries mapping parameter names to type annotations.')
    checkers = []
    param_names = list(api_signature.parameters)
    for param_name, param_type in signature.items():
        if isinstance(param_name, int) and param_name < len(api_signature.parameters):
            param_name = list(api_signature.parameters.values())[param_name].name
        param = api_signature.parameters.get(param_name, None)
        if param is None:
            raise ValueError(f'signature includes annotation for unknown parameter {param_name!r}.')
        if param.kind not in (tf_inspect.Parameter.POSITIONAL_ONLY, tf_inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise ValueError(f"Dispatch currently only supports type annotations for positional parameters; can't handle annotation for {param.kind!r} parameter {param_name}.")
        checker = make_type_checker(param_type)
        index = param_names.index(param_name)
        checkers.append((index, checker))
    return _api_dispatcher.PySignatureChecker(checkers)