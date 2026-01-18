import os
import sys
import threading
from . import process
from . import reduction
class SpawnProcess(process.BaseProcess):
    _start_method = 'spawn'

    @staticmethod
    def _Popen(process_obj):
        from .popen_spawn_win32 import Popen
        return Popen(process_obj)

    @staticmethod
    def _after_fork():
        pass