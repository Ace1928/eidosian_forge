import functools
import re
import types
def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__
    lines = output_docstring.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*(Args|Parameters):\\s*$', lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = '\n'.join(lines[i + 1:])
        params_docstring = _convert_output_args_doc(params_docstring)
    else:
        raise ValueError(f'No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has docstring and contain either `Args` or `Parameters`.')
    full_output_type = f'{output_type.__module__}.{output_type.__name__}'
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith('TF') else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    result = intro + params_docstring
    if min_indent is not None:
        lines = result.split('\n')
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        if indent < min_indent:
            to_add = ' ' * (min_indent - indent)
            lines = [f'{to_add}{line}' if len(line) > 0 else line for line in lines]
            result = '\n'.join(lines)
    return result