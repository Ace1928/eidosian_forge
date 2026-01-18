from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
class MockRootConfig(object):

    def __init__(self, fence_pattern, ruler_pattern):
        self.markup = MockMarkupConfig(fence_pattern, ruler_pattern)