import doctest
import sys
from os.path import abspath, dirname, join
class CustomOutputChecker(OutputChecker):

    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)