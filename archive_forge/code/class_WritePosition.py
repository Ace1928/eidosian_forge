from __future__ import unicode_literals
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from collections import defaultdict, namedtuple
class WritePosition(object):

    def __init__(self, xpos, ypos, width, height, extended_height=None):
        assert height >= 0
        assert extended_height is None or extended_height >= 0
        assert width >= 0
        self.xpos = xpos
        self.ypos = ypos
        self.width = width
        self.height = height
        self.extended_height = extended_height or height

    def __repr__(self):
        return '%s(%r, %r, %r, %r, %r)' % (self.__class__.__name__, self.xpos, self.ypos, self.width, self.height, self.extended_height)