import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def ProcessVariablesAndConditionsInDict(the_dict, phase, variables_in, build_file, the_dict_key=None):
    """Handle all variable and command expansion and conditional evaluation.

  This function is the public entry point for all variable expansions and
  conditional evaluations.  The variables_in dictionary will not be modified
  by this function.
  """
    variables = variables_in.copy()
    LoadAutomaticVariablesFromDict(variables, the_dict)
    if 'variables' in the_dict:
        for key, value in the_dict['variables'].items():
            variables[key] = value
        ProcessVariablesAndConditionsInDict(the_dict['variables'], phase, variables, build_file, 'variables')
    LoadVariablesFromVariablesDict(variables, the_dict, the_dict_key)
    for key, value in the_dict.items():
        if key != 'variables' and type(value) is str:
            expanded = ExpandVariables(value, phase, variables, build_file)
            if type(expanded) not in (str, int):
                raise ValueError('Variable expansion in this context permits str and int ' + 'only, found ' + expanded.__class__.__name__ + ' for ' + key)
            the_dict[key] = expanded
    variables = variables_in.copy()
    LoadAutomaticVariablesFromDict(variables, the_dict)
    LoadVariablesFromVariablesDict(variables, the_dict, the_dict_key)
    ProcessConditionsInDict(the_dict, phase, variables, build_file)
    variables = variables_in.copy()
    LoadAutomaticVariablesFromDict(variables, the_dict)
    LoadVariablesFromVariablesDict(variables, the_dict, the_dict_key)
    for key, value in the_dict.items():
        if key == 'variables' or type(value) is str:
            continue
        if type(value) is dict:
            ProcessVariablesAndConditionsInDict(value, phase, variables, build_file, key)
        elif type(value) is list:
            ProcessVariablesAndConditionsInList(value, phase, variables, build_file)
        elif type(value) is not int:
            raise TypeError('Unknown type ' + value.__class__.__name__ + ' for ' + key)