from ray.dag import DAGNode
from ray.util.annotations import DeveloperAPI
def _get_other_args_to_resolve_lines(other_args_to_resolve):
    if not other_args_to_resolve:
        return '{}'
    indent = _get_indentation()
    other_args_to_resolve_lines = []
    for key, val in other_args_to_resolve.items():
        if isinstance(val, DAGNode):
            node_repr_lines = str(val).split('\n')
            for index, node_repr_line in enumerate(node_repr_lines):
                if index == 0:
                    other_args_to_resolve_lines.append(f'{indent}{key}:' + f'{indent}' + '\n' + f'{indent}{indent}{indent}' + node_repr_line)
                else:
                    other_args_to_resolve_lines.append(f'{indent}{indent}' + node_repr_line)
        else:
            other_args_to_resolve_lines.append(f'{indent}{key}: ' + str(val))
    other_args_to_resolve_line = '{'
    for line in other_args_to_resolve_lines:
        other_args_to_resolve_line += f'\n{indent}{line}'
    other_args_to_resolve_line += f'\n{indent}}}'
    return other_args_to_resolve_line