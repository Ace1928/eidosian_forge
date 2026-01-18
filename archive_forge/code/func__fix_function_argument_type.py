import ast
from .qt import ClassFlag, qt_class_flags
def _fix_function_argument_type(type, for_return):
    """Fix function argument/return qualifiers using some heuristics for Qt."""
    if type == 'float':
        return 'double'
    if type == 'str':
        type = 'QString'
    if not type.startswith('Q'):
        return type
    flags = qt_class_flags(type)
    if flags & ClassFlag.PASS_BY_VALUE:
        return type
    if flags & ClassFlag.PASS_BY_CONSTREF:
        return type if for_return else f'const {type} &'
    if flags & ClassFlag.PASS_BY_REF:
        return type if for_return else f'{type} &'
    return type + ' *'