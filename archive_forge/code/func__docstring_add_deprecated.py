import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _docstring_add_deprecated(func, kwarg_mapping, deprecated_version):
    """Add deprecated kwarg(s) to the "Other Params" section of a docstring.

    Parameters
    ----------
    func : function
        The function whose docstring we wish to update.
    kwarg_mapping : dict
        A dict containing {old_arg: new_arg} key/value pairs, see
        `deprecate_parameter`.
    deprecated_version : str
        A major.minor version string specifying when old_arg was
        deprecated.

    Returns
    -------
    new_doc : str
        The updated docstring. Returns the original docstring if numpydoc is
        not available.
    """
    if func.__doc__ is None:
        return None
    try:
        from numpydoc.docscrape import FunctionDoc, Parameter
    except ImportError:
        return func.__doc__
    Doc = FunctionDoc(func)
    for old_arg, new_arg in kwarg_mapping.items():
        desc = []
        if new_arg is None:
            desc.append(f'`{old_arg}` is deprecated.')
        else:
            desc.append(f'Deprecated in favor of `{new_arg}`.')
        desc += ['', f'.. deprecated:: {deprecated_version}']
        Doc['Other Parameters'].append(Parameter(name=old_arg, type='DEPRECATED', desc=desc))
    new_docstring = str(Doc)
    split = new_docstring.split('\n')
    no_header = split[1:]
    while not no_header[0].strip():
        no_header.pop(0)
    descr = no_header.pop(0)
    while no_header[0].strip():
        descr += '\n    ' + no_header.pop(0)
    descr += '\n\n'
    final_docstring = descr + '\n    '.join(no_header)
    final_docstring = '\n'.join([line.rstrip() for line in final_docstring.split('\n')])
    return final_docstring