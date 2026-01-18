import re
import sys
import unittest
from doctest import DocTestFinder, DocTestRunner, TestResults
from IPython.terminal.interactiveshell import InteractiveShell
class IPython2PythonConverter(object):
    """Convert IPython 'syntax' to valid Python.

    Eventually this code may grow to be the full IPython syntax conversion
    implementation, but for now it only does prompt conversion."""

    def __init__(self):
        self.rps1 = re.compile('In\\ \\[\\d+\\]: ')
        self.rps2 = re.compile('\\ \\ \\ \\.\\.\\.+: ')
        self.rout = re.compile('Out\\[\\d+\\]: \\s*?\\n?')
        self.pyps1 = '>>> '
        self.pyps2 = '... '
        self.rpyps1 = re.compile('(\\s*%s)(.*)$' % self.pyps1)
        self.rpyps2 = re.compile('(\\s*%s)(.*)$' % self.pyps2)

    def __call__(self, ds):
        """Convert IPython prompts to python ones in a string."""
        from . import globalipapp
        pyps1 = '>>> '
        pyps2 = '... '
        pyout = ''
        dnew = ds
        dnew = self.rps1.sub(pyps1, dnew)
        dnew = self.rps2.sub(pyps2, dnew)
        dnew = self.rout.sub(pyout, dnew)
        ip = InteractiveShell.instance()
        out = []
        newline = out.append
        for line in dnew.splitlines():
            mps1 = self.rpyps1.match(line)
            if mps1 is not None:
                prompt, text = mps1.groups()
                newline(prompt + ip.prefilter(text, False))
                continue
            mps2 = self.rpyps2.match(line)
            if mps2 is not None:
                prompt, text = mps2.groups()
                newline(prompt + ip.prefilter(text, True))
                continue
            newline(line)
        newline('')
        return '\n'.join(out)