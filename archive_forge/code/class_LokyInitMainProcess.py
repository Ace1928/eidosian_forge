import sys
from multiprocessing.context import assert_spawning
from multiprocessing.process import BaseProcess
class LokyInitMainProcess(LokyProcess):
    _start_method = 'loky_init_main'

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, daemon=None):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon, init_main_module=True)