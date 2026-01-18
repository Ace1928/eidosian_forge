import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _build_node_error_message(op):
    """Returns the formatted error message for the given op.

  Args:
    op: The node.

  Returns:
    The formatted error message for the given op with traceback.
  """
    node_error_message = [f'Detected at node {op.name!r} defined at (most recent call last):']
    field_dict = _compute_field_dict(op)
    for frame in field_dict['definition_traceback']:
        if '<embedded' not in frame:
            node_error_message.extend([f'  {line}' for line in frame.split('\n') if line.strip()])
    node_error_message.append(f'Node: {op.name!r}')
    return '\n'.join(node_error_message)