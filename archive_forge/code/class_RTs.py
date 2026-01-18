import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
class RTs:
    """
    A namespace for the refactoring tools. This avoids creating these at
    the module level, which slows down the module import. (See issue #117).

    There are two possible grammars: with or without the print statement.
    Hence we have two possible refactoring tool implementations.
    """
    _rt = None
    _rtp = None
    _rt_py2_detect = None
    _rtp_py2_detect = None

    @staticmethod
    def setup():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt, RTs._rtp]:
            RTs._rt = RefactoringTool(myfixes)
            RTs._rtp = RefactoringTool(myfixes, {'print_function': True})

    @staticmethod
    def setup_detect_python2():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt_py2_detect, RTs._rtp_py2_detect]:
            RTs._rt_py2_detect = RefactoringTool(py2_detect_fixers)
            RTs._rtp_py2_detect = RefactoringTool(py2_detect_fixers, {'print_function': True})