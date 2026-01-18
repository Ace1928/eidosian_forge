import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _cmdloop(self):
    while True:
        try:
            self.allow_kbdint = True
            self.cmdloop()
            self.allow_kbdint = False
            break
        except KeyboardInterrupt:
            self.message('--KeyboardInterrupt--')