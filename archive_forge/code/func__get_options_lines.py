from ray.dag import DAGNode
from ray.util.annotations import DeveloperAPI
def _get_options_lines(bound_options):
    """Pretty prints .options() in DAGNode. Only prints non-empty values."""
    if not bound_options:
        return '{}'
    indent = _get_indentation()
    options_lines = []
    for key, val in bound_options.items():
        if val:
            options_lines.append(f'{indent}{key}: ' + str(val))
    options_line = '{'
    for line in options_lines:
        options_line += f'\n{indent}{line}'
    options_line += f'\n{indent}}}'
    return options_line