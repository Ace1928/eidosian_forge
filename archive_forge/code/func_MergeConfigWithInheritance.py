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
def MergeConfigWithInheritance(new_configuration_dict, build_file, target_dict, configuration, visited):
    if configuration in visited:
        return
    configuration_dict = target_dict['configurations'][configuration]
    for parent in configuration_dict.get('inherit_from', []):
        MergeConfigWithInheritance(new_configuration_dict, build_file, target_dict, parent, visited + [configuration])
    MergeDicts(new_configuration_dict, configuration_dict, build_file, build_file)
    if 'abstract' in new_configuration_dict:
        del new_configuration_dict['abstract']