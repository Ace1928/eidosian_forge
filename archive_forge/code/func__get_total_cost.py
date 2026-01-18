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
def _get_total_cost(self, aggregated_profile, cost_type):
    if cost_type == 'exec_time':
        return aggregated_profile.total_exec_time
    elif cost_type == 'op_time':
        return aggregated_profile.total_op_time
    else:
        raise ValueError('Unsupported cost type: %s' % cost_type)