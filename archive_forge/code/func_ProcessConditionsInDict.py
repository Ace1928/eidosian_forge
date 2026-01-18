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
def ProcessConditionsInDict(the_dict, phase, variables, build_file):
    if phase == PHASE_EARLY:
        conditions_key = 'conditions'
    elif phase == PHASE_LATE:
        conditions_key = 'target_conditions'
    elif phase == PHASE_LATELATE:
        return
    else:
        assert False
    if conditions_key not in the_dict:
        return
    conditions_list = the_dict[conditions_key]
    del the_dict[conditions_key]
    for condition in conditions_list:
        merge_dict = EvalCondition(condition, conditions_key, phase, variables, build_file)
        if merge_dict is not None:
            ProcessVariablesAndConditionsInDict(merge_dict, phase, variables, build_file)
            MergeDicts(the_dict, merge_dict, build_file, build_file)