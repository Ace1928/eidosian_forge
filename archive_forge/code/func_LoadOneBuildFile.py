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
def LoadOneBuildFile(build_file_path, data, aux_data, includes, is_target, check):
    if build_file_path in data:
        return data[build_file_path]
    if os.path.exists(build_file_path):
        build_file_contents = open(build_file_path, encoding='utf-8').read()
    else:
        raise GypError(f'{build_file_path} not found (cwd: {os.getcwd()})')
    build_file_data = None
    try:
        if check:
            build_file_data = CheckedEval(build_file_contents)
        else:
            build_file_data = eval(build_file_contents, {'__builtins__': {}}, None)
    except SyntaxError as e:
        e.filename = build_file_path
        raise
    except Exception as e:
        gyp.common.ExceptionAppend(e, 'while reading ' + build_file_path)
        raise
    if type(build_file_data) is not dict:
        raise GypError('%s does not evaluate to a dictionary.' % build_file_path)
    data[build_file_path] = build_file_data
    aux_data[build_file_path] = {}
    if 'skip_includes' not in build_file_data or not build_file_data['skip_includes']:
        try:
            if is_target:
                LoadBuildFileIncludesIntoDict(build_file_data, build_file_path, data, aux_data, includes, check)
            else:
                LoadBuildFileIncludesIntoDict(build_file_data, build_file_path, data, aux_data, None, check)
        except Exception as e:
            gyp.common.ExceptionAppend(e, 'while reading includes of ' + build_file_path)
            raise
    return build_file_data