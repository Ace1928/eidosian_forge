import argparse
import os
import re
import numpy as np
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import profiling
from tensorflow.python.debug.lib import source_utils
def _get_list_profile_lines(self, device_name, device_index, device_count, profile_datum_list, sort_by, sort_reverse, time_unit, device_name_filter=None, node_name_filter=None, op_type_filter=None, screen_cols=80):
    """Get `RichTextLines` object for list_profile command for a given device.

    Args:
      device_name: (string) Device name.
      device_index: (int) Device index.
      device_count: (int) Number of devices.
      profile_datum_list: List of `ProfileDatum` objects.
      sort_by: (string) Identifier of column to sort. Sort identifier
          must match value of SORT_OPS_BY_OP_NAME, SORT_OPS_BY_OP_TYPE,
          SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_MEMORY or SORT_OPS_BY_LINE.
      sort_reverse: (bool) Whether to sort in descending instead of default
          (ascending) order.
      time_unit: time unit, must be in cli_shared.TIME_UNITS.
      device_name_filter: Regular expression to filter by device name.
      node_name_filter: Regular expression to filter by node name.
      op_type_filter: Regular expression to filter by op type.
      screen_cols: (int) Number of columns available on the screen (i.e.,
        available screen width).

    Returns:
      `RichTextLines` object containing a table that displays profiling
      information for each op.
    """
    profile_data = ProfileDataTableView(profile_datum_list, time_unit=time_unit)
    total_op_time = sum((datum.op_time for datum in profile_datum_list))
    total_exec_time = sum((datum.node_exec_stats.all_end_rel_micros for datum in profile_datum_list))
    device_total_row = ['Device Total', '', cli_shared.time_to_readable_str(total_op_time, force_time_unit=time_unit), cli_shared.time_to_readable_str(total_exec_time, force_time_unit=time_unit)]
    column_widths = [len(column_name) for column_name in profile_data.column_names()]
    for col in range(len(device_total_row)):
        column_widths[col] = max(column_widths[col], len(device_total_row[col]))
    for col in range(len(column_widths)):
        for row in range(profile_data.row_count()):
            column_widths[col] = max(column_widths[col], len(profile_data.value(row, col, device_name_filter=device_name_filter, node_name_filter=node_name_filter, op_type_filter=op_type_filter)))
        column_widths[col] += 2
    output = [RL('-' * screen_cols)]
    device_row = 'Device %d of %d: %s' % (device_index + 1, device_count, device_name)
    output.append(RL(device_row))
    output.append(RL())
    base_command = 'list_profile'
    row = RL()
    for col in range(profile_data.column_count()):
        column_name = profile_data.column_names()[col]
        sort_id = profile_data.column_sort_id(col)
        command = '%s -s %s' % (base_command, sort_id)
        if sort_by == sort_id and (not sort_reverse):
            command += ' -r'
        head_menu_item = debugger_cli_common.MenuItem(None, command)
        row += RL(column_name, font_attr=[head_menu_item, 'bold'])
        row += RL(' ' * (column_widths[col] - len(column_name)))
    output.append(row)
    for row in range(profile_data.row_count()):
        new_row = RL()
        for col in range(profile_data.column_count()):
            new_cell = profile_data.value(row, col, device_name_filter=device_name_filter, node_name_filter=node_name_filter, op_type_filter=op_type_filter)
            new_row += new_cell
            new_row += RL(' ' * (column_widths[col] - len(new_cell)))
        output.append(new_row)
    row_str = ''
    for width, row in zip(column_widths, device_total_row):
        row_str += ('{:<%d}' % width).format(row)
    output.append(RL())
    output.append(RL(row_str))
    return debugger_cli_common.rich_text_lines_from_rich_line_list(output)