import os
import sys
import threading
from . import process
from . import reduction
class ForkProcess(process.BaseProcess):
    _start_method = 'fork'

    @staticmethod
    def _Popen(process_obj):
        from .popen_fork import Popen
        return Popen(process_obj)