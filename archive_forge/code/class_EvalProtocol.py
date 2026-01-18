import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
class EvalProtocol(basic.LineOnlyReceiver):
    delimiter = '\n'

    def __init__(self, myrepl):
        self.repl = myrepl

    def lineReceived(self, line):
        self.repl.main_loop.process_input(line)
        self.repl.main_loop.process_input(['enter'])