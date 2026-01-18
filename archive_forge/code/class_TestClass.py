import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestClass:
    """Just a simple test class instrumented for the test cases"""
    class_member = 'class_member'

    @staticmethod
    def use_actions(actions):
        TestClass.actions = actions

    def __init__(self):
        TestClass.actions.append('init')

    def foo(self, x):
        TestClass.actions.append(('foo', x))
        return 'foo'