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
def LoadTargetBuildFilesParallel(build_files, data, variables, includes, depth, check, generator_input_info):
    parallel_state = ParallelState()
    parallel_state.condition = threading.Condition()
    parallel_state.dependencies = list(build_files)
    parallel_state.scheduled = set(build_files)
    parallel_state.pending = 0
    parallel_state.data = data
    try:
        parallel_state.condition.acquire()
        while parallel_state.dependencies or parallel_state.pending:
            if parallel_state.error:
                break
            if not parallel_state.dependencies:
                parallel_state.condition.wait()
                continue
            dependency = parallel_state.dependencies.pop()
            parallel_state.pending += 1
            global_flags = {'path_sections': globals()['path_sections'], 'non_configuration_keys': globals()['non_configuration_keys'], 'multiple_toolsets': globals()['multiple_toolsets']}
            if not parallel_state.pool:
                parallel_state.pool = multiprocessing.Pool(multiprocessing.cpu_count())
            parallel_state.pool.apply_async(CallLoadTargetBuildFile, args=(global_flags, dependency, variables, includes, depth, check, generator_input_info), callback=parallel_state.LoadTargetBuildFileCallback)
    except KeyboardInterrupt as e:
        parallel_state.pool.terminate()
        raise e
    parallel_state.condition.release()
    parallel_state.pool.close()
    parallel_state.pool.join()
    parallel_state.pool = None
    if parallel_state.error:
        sys.exit(1)