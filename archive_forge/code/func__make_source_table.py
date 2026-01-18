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
def _make_source_table(self, source_list, is_tf_py_library):
    """Make a table summarizing the source files that create nodes and tensors.

    Args:
      source_list: List of source files and related information as a list of
        tuples (file_path, is_tf_library, num_nodes, num_tensors, num_dumps,
        first_line).
      is_tf_py_library: (`bool`) whether this table is for files that belong
        to the TensorFlow Python library.

    Returns:
      The table as a `debugger_cli_common.RichTextLines` object.
    """
    path_head = 'Source file path'
    num_nodes_head = '#(nodes)'
    num_tensors_head = '#(tensors)'
    num_dumps_head = '#(tensor dumps)'
    if is_tf_py_library:
        color = cli_shared.COLOR_GRAY
        lines = [RL('TensorFlow Python library file(s):', color)]
    else:
        color = cli_shared.COLOR_WHITE
        lines = [RL('File(s) outside TensorFlow Python library:', color)]
    if not source_list:
        lines.append(RL('[No files.]'))
        lines.append(RL())
        return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)
    path_column_width = max(max((len(item[0]) for item in source_list)), len(path_head)) + 1
    num_nodes_column_width = max(max((len(str(item[2])) for item in source_list)), len(num_nodes_head)) + 1
    num_tensors_column_width = max(max((len(str(item[3])) for item in source_list)), len(num_tensors_head)) + 1
    head = RL(path_head + ' ' * (path_column_width - len(path_head)), color)
    head += RL(num_nodes_head + ' ' * (num_nodes_column_width - len(num_nodes_head)), color)
    head += RL(num_tensors_head + ' ' * (num_tensors_column_width - len(num_tensors_head)), color)
    head += RL(num_dumps_head, color)
    lines.append(head)
    for file_path, _, num_nodes, num_tensors, num_dumps, first_line_num in source_list:
        path_attributes = [color]
        if source_utils.is_extension_uncompiled_python_source(file_path):
            path_attributes.append(debugger_cli_common.MenuItem(None, 'ps %s -b %d' % (file_path, first_line_num)))
        line = RL(file_path, path_attributes)
        line += ' ' * (path_column_width - len(line))
        line += RL(str(num_nodes) + ' ' * (num_nodes_column_width - len(str(num_nodes))), color)
        line += RL(str(num_tensors) + ' ' * (num_tensors_column_width - len(str(num_tensors))), color)
        line += RL(str(num_dumps), color)
        lines.append(line)
    lines.append(RL())
    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)