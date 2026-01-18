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
def _format_neighbors(self, neighbor_type, non_ctrls, ctrls):
    """List neighbors (inputs or recipients) of a node.

    Args:
      neighbor_type: ("input" | "recipient")
      non_ctrls: Non-control neighbor node names, as a list of str.
      ctrls: Control neighbor node names, as a list of str.

    Returns:
      A RichTextLines object.
    """
    lines = []
    font_attr_segs = {}
    lines.append('')
    lines.append('  %d %s(s) + %d control %s(s):' % (len(non_ctrls), neighbor_type, len(ctrls), neighbor_type))
    lines.append('    %d %s(s):' % (len(non_ctrls), neighbor_type))
    for non_ctrl in non_ctrls:
        line = '      [%s] %s' % (self._debug_dump.node_op_type(non_ctrl), non_ctrl)
        lines.append(line)
        font_attr_segs[len(lines) - 1] = [(len(line) - len(non_ctrl), len(line), debugger_cli_common.MenuItem(None, 'ni -a -d -t %s' % non_ctrl))]
    if ctrls:
        lines.append('')
        lines.append('    %d control %s(s):' % (len(ctrls), neighbor_type))
        for ctrl in ctrls:
            line = '      [%s] %s' % (self._debug_dump.node_op_type(ctrl), ctrl)
            lines.append(line)
            font_attr_segs[len(lines) - 1] = [(len(line) - len(ctrl), len(line), debugger_cli_common.MenuItem(None, 'ni -a -d -t %s' % ctrl))]
    return debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)