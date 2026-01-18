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
class ProfileDataTableView(object):
    """Table View of profiling data."""

    def __init__(self, profile_datum_list, time_unit=cli_shared.TIME_UNIT_US):
        """Constructor.

    Args:
      profile_datum_list: List of `ProfileDatum` objects.
      time_unit: must be in cli_shared.TIME_UNITS.
    """
        self._profile_datum_list = profile_datum_list
        self.formatted_start_time = [datum.start_time for datum in profile_datum_list]
        self.formatted_op_time = [cli_shared.time_to_readable_str(datum.op_time, force_time_unit=time_unit) for datum in profile_datum_list]
        self.formatted_exec_time = [cli_shared.time_to_readable_str(datum.node_exec_stats.all_end_rel_micros, force_time_unit=time_unit) for datum in profile_datum_list]
        self._column_names = ['Node', 'Op Type', 'Start Time (us)', 'Op Time (%s)' % time_unit, 'Exec Time (%s)' % time_unit, 'Filename:Lineno(function)']
        self._column_sort_ids = [SORT_OPS_BY_OP_NAME, SORT_OPS_BY_OP_TYPE, SORT_OPS_BY_START_TIME, SORT_OPS_BY_OP_TIME, SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_LINE]

    def value(self, row, col, device_name_filter=None, node_name_filter=None, op_type_filter=None):
        """Get the content of a cell of the table.

    Args:
      row: (int) row index.
      col: (int) column index.
      device_name_filter: Regular expression to filter by device name.
      node_name_filter: Regular expression to filter by node name.
      op_type_filter: Regular expression to filter by op type.

    Returns:
      A debuggre_cli_common.RichLine object representing the content of the
      cell, potentially with a clickable MenuItem.

    Raises:
      IndexError: if row index is out of range.
    """
        menu_item = None
        if col == 0:
            text = self._profile_datum_list[row].node_exec_stats.node_name
        elif col == 1:
            text = self._profile_datum_list[row].op_type
        elif col == 2:
            text = str(self.formatted_start_time[row])
        elif col == 3:
            text = str(self.formatted_op_time[row])
        elif col == 4:
            text = str(self.formatted_exec_time[row])
        elif col == 5:
            command = 'ps'
            if device_name_filter:
                command += ' --%s %s' % (_DEVICE_NAME_FILTER_FLAG, device_name_filter)
            if node_name_filter:
                command += ' --%s %s' % (_NODE_NAME_FILTER_FLAG, node_name_filter)
            if op_type_filter:
                command += ' --%s %s' % (_OP_TYPE_FILTER_FLAG, op_type_filter)
            command += ' %s --init_line %d' % (self._profile_datum_list[row].file_path, self._profile_datum_list[row].line_number)
            menu_item = debugger_cli_common.MenuItem(None, command)
            text = self._profile_datum_list[row].file_line_func
        else:
            raise IndexError('Invalid column index %d.' % col)
        return RL(text, font_attr=menu_item)

    def row_count(self):
        return len(self._profile_datum_list)

    def column_count(self):
        return len(self._column_names)

    def column_names(self):
        return self._column_names

    def column_sort_id(self, col):
        return self._column_sort_ids[col]