from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _GetOptsAssignmentTemplate(command):
    if command == name:
        return opts_assignment_main_command_template
    else:
        return opts_assignment_subcommand_template