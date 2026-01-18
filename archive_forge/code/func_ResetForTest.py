from mx import DateTime
from __future__ import absolute_import
from __future__ import print_function
import os
import pdb
import sys
import traceback
from absl import app
from absl import flags
def ResetForTest():
    """Reset the module for test purposes; ONLY use for testing."""
    global _cmd_argv, _cmd_list, _cmd_alias_list, _cmd_default, _cmd_gnugetopt
    _cmd_argv = None
    _cmd_list = {}
    _cmd_alias_list = {}
    _cmd_default = 'help'
    use_gnu_opt_name = 'APPCOMMAND_USE_GNU_GET_OPT_FOR_SUBCOMMAND'
    if use_gnu_opt_name in os.environ:
        _cmd_gnugetopt = os.environ[use_gnu_opt_name] == '1'
    else:
        _cmd_gnugetopt = False
    AddCmd('help', _CmdHelp)