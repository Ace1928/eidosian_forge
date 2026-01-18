import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
class CallbackInstructionTest(GraphicUnitTest):

    def test_from_kv(self):
        from textwrap import dedent
        from kivy.lang import Builder
        root = Builder.load_string(dedent("        Widget:\n            canvas:\n                Callback:\n                    callback: lambda __: setattr(self, 'callback_test', 'TEST')\n        "))
        r = self.render
        r(root)
        self.assertTrue(root.callback_test == 'TEST')