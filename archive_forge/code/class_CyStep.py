from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyStep(CythonExecutionControlCommand, libpython.PythonStepperMixin):
    """Step through Cython, Python or C code."""
    name = 'cy -step'
    stepinto = True

    @libpython.dont_suppress_errors
    def invoke(self, args, from_tty):
        if self.is_python_function():
            self.python_step(self.stepinto)
        elif not self.is_cython_function():
            if self.stepinto:
                command = 'step'
            else:
                command = 'next'
            self.finish_executing(gdb.execute(command, to_string=True))
        else:
            self.step(stepinto=self.stepinto)