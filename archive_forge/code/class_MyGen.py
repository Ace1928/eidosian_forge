import functools
import inspect
import opcode
import os
import sys
import traceback
import types
from . import events
from . import futures
from .log import logger
class MyGen:

    def __init__(self):
        self.send_args = None

    def __iter__(self):
        return self

    def __next__(self):
        return 42

    def send(self, *what):
        self.send_args = what
        return None