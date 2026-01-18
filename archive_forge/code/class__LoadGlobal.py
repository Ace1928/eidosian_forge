import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _LoadGlobal(_BaseLoadHandler):
    opname = 'LOAD_GLOBAL'