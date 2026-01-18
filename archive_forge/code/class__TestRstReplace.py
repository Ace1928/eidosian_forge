import unittest
from kivy.tests.common import GraphicUnitTest
class _TestRstReplace(RstDocument):

    def __init__(self, **kwargs):
        super(_TestRstReplace, self).__init__(**kwargs)
        self.text = '\n    .. |uni| unicode:: 0xe4\n    .. |nbsp| unicode:: 0xA0\n    .. |text| replace:: is\n    .. |hop| replace:: replaced\n    .. _hop: https://kivy.org\n\n    |uni| |nbsp| |text| |hop|_\n    '