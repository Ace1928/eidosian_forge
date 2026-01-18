import warnings
from warnings import warn
import breezy
def _decorate_docstring(callable, deprecation_version, label, decorated_callable):
    if callable.__doc__:
        docstring_lines = callable.__doc__.split('\n')
    else:
        docstring_lines = []
    if len(docstring_lines) == 0:
        decorated_callable.__doc__ = deprecation_version % ('This ' + label)
    elif len(docstring_lines) == 1:
        decorated_callable.__doc__ = callable.__doc__ + '\n' + '\n' + deprecation_version % ('This ' + label) + '\n'
    else:
        spaces = len(docstring_lines[-1])
        new_doc = callable.__doc__
        new_doc += '\n' + ' ' * spaces
        new_doc += deprecation_version % ('This ' + label)
        new_doc += '\n' + ' ' * spaces
        decorated_callable.__doc__ = new_doc