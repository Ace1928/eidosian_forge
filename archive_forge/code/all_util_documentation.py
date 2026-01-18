import re as _re
import sys as _sys
from tensorflow.python.util import tf_inspect as _tf_inspect
Removes symbols in a module that are not referenced by a docstring.

  Args:
    module_name: the name of the module (usually `__name__`).
    allowed_exception_list: a list of names that should not be removed.
    doc_string_modules: a list of modules from which to take the docstrings.
    If None, then a list containing only the module named `module_name` is used.

    Furthermore, if a symbol previously added with `add_to_global_allowlist`,
    then it will always be allowed. This is useful for internal tests.

  Returns:
    None
  