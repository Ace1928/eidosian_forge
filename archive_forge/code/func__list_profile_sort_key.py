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
def _list_profile_sort_key(profile_datum, sort_by):
    """Get a profile_datum property to sort by in list_profile command.

  Args:
    profile_datum: A `ProfileDatum` object.
    sort_by: (string) indicates a value to sort by.
      Must be one of SORT_BY* constants.

  Returns:
    profile_datum property to sort by.
  """
    if sort_by == SORT_OPS_BY_OP_NAME:
        return profile_datum.node_exec_stats.node_name
    elif sort_by == SORT_OPS_BY_OP_TYPE:
        return profile_datum.op_type
    elif sort_by == SORT_OPS_BY_LINE:
        return profile_datum.file_line_func
    elif sort_by == SORT_OPS_BY_OP_TIME:
        return profile_datum.op_time
    elif sort_by == SORT_OPS_BY_EXEC_TIME:
        return profile_datum.node_exec_stats.all_end_rel_micros
    else:
        return profile_datum.node_exec_stats.all_start_micros