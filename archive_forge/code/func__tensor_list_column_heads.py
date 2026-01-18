import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def _tensor_list_column_heads(self, parsed, max_timestamp_width, max_dump_size_width, max_op_type_width):
    """Generate a line containing the column heads of the tensor list.

    Args:
      parsed: Parsed arguments (by argparse) of the list_tensors command.
      max_timestamp_width: (int) maximum width of the timestamp column.
      max_dump_size_width: (int) maximum width of the dump size column.
      max_op_type_width: (int) maximum width of the op type column.

    Returns:
      A RichTextLines object.
    """
    base_command = 'list_tensors'
    if parsed.tensor_filter:
        base_command += ' -f %s' % parsed.tensor_filter
    if parsed.op_type_filter:
        base_command += ' -t %s' % parsed.op_type_filter
    if parsed.node_name_filter:
        base_command += ' -n %s' % parsed.node_name_filter
    attr_segs = {0: []}
    row = self._TIMESTAMP_COLUMN_HEAD
    command = '%s -s %s' % (base_command, SORT_TENSORS_BY_TIMESTAMP)
    if parsed.sort_by == SORT_TENSORS_BY_TIMESTAMP and (not parsed.reverse):
        command += ' -r'
    attr_segs[0].append((0, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
    row += ' ' * (max_timestamp_width - len(row))
    prev_len = len(row)
    row += self._DUMP_SIZE_COLUMN_HEAD
    command = '%s -s %s' % (base_command, SORT_TENSORS_BY_DUMP_SIZE)
    if parsed.sort_by == SORT_TENSORS_BY_DUMP_SIZE and (not parsed.reverse):
        command += ' -r'
    attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
    row += ' ' * (max_dump_size_width + max_timestamp_width - len(row))
    prev_len = len(row)
    row += self._OP_TYPE_COLUMN_HEAD
    command = '%s -s %s' % (base_command, SORT_TENSORS_BY_OP_TYPE)
    if parsed.sort_by == SORT_TENSORS_BY_OP_TYPE and (not parsed.reverse):
        command += ' -r'
    attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
    row += ' ' * (max_op_type_width + max_dump_size_width + max_timestamp_width - len(row))
    prev_len = len(row)
    row += self._TENSOR_NAME_COLUMN_HEAD
    command = '%s -s %s' % (base_command, SORT_TENSORS_BY_TENSOR_NAME)
    if parsed.sort_by == SORT_TENSORS_BY_TENSOR_NAME and (not parsed.reverse):
        command += ' -r'
    attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem('', command), 'bold']))
    row += ' ' * (max_op_type_width + max_dump_size_width + max_timestamp_width - len(row))
    return debugger_cli_common.RichTextLines([row], font_attr_segs=attr_segs)