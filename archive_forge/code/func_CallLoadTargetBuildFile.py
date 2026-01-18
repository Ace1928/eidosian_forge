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
def CallLoadTargetBuildFile(global_flags, build_file_path, variables, includes, depth, check, generator_input_info):
    """Wrapper around LoadTargetBuildFile for parallel processing.

     This wrapper is used when LoadTargetBuildFile is executed in
     a worker process.
  """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for key, value in global_flags.items():
            globals()[key] = value
        SetGeneratorGlobals(generator_input_info)
        result = LoadTargetBuildFile(build_file_path, per_process_data, per_process_aux_data, variables, includes, depth, check, False)
        if not result:
            return result
        build_file_path, dependencies = result
        build_file_data = per_process_data.pop(build_file_path)
        return (build_file_path, build_file_data, dependencies)
    except GypError as e:
        sys.stderr.write('gyp: %s\n' % e)
        return None
    except Exception as e:
        print('Exception:', e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None