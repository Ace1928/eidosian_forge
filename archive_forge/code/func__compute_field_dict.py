import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _compute_field_dict(op):
    """Return a dictionary mapping interpolation tokens to values.

  Args:
    op: op.Operation object.

  Returns:
    A dictionary mapping string tokens to string values.  The keys are shown
    below along with example values.
    {
      "file": "tool_utils.py",
      "lineno": "124",
      "line": "  source code line",
      "defined_at": " (defined at tool_utils.py:124)",
      "colocations":
          '''Node-device colocations active during op creation:
               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>
               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>'''
      "devices":
          '''Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
      "devs_and_colocs": A concatenation of colocations and devices, e.g.
          '''Node-device colocations active during op creation:
               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>
               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>'''
             Device assignments active during op 'foo' creation:
               with tf.device(/cpu:0): <test_1.py:27>
               with tf.device(some_func<foo.py, 123>): <test_2.py:38>'''
    }
  """
    colocation_summary = _compute_colocation_summary_from_op(op)
    device_summary = _compute_device_assignment_summary_from_op(op)
    combined_summary = '\n'.join([colocation_summary, device_summary])
    if op.traceback is None:
        filename = '<unknown>'
        definition_traceback = ''
        lineno = 0
        line = ''
        defined_at = '<unknown>'
    else:
        frame = op.traceback.last_user_frame()
        filename = frame.filename
        definition_traceback = traceback.format_list(op.traceback.get_user_frames())
        lineno = frame.lineno
        line = frame.line
        defined_at = f'{filename}:{lineno:d}'
    field_dict = {'colocations': colocation_summary, 'devices': device_summary, 'devs_and_colocs': combined_summary, 'defined_at': defined_at, 'file': filename, 'lineno': lineno, 'line': line, 'definition_traceback': definition_traceback}
    return field_dict