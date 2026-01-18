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
def ValidateTargetType(target, target_dict):
    """Ensures the 'type' field on the target is one of the known types.

  Arguments:
    target: string, name of target.
    target_dict: dict, target spec.

  Raises an exception on error.
  """
    VALID_TARGET_TYPES = ('executable', 'loadable_module', 'static_library', 'shared_library', 'mac_kernel_extension', 'none', 'windows_driver')
    target_type = target_dict.get('type', None)
    if target_type not in VALID_TARGET_TYPES:
        raise GypError("Target %s has an invalid target type '%s'.  Must be one of %s." % (target, target_type, '/'.join(VALID_TARGET_TYPES)))
    if target_dict.get('standalone_static_library', 0) and (not target_type == 'static_library'):
        raise GypError('Target %s has type %s but standalone_static_library flag is only valid for static_library type.' % (target, target_type))