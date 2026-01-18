from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class TypedProperties(object):
    """Test class for testing Python Fire with properties of various types."""

    def __init__(self):
        self.alpha = True
        self.beta = (1, 2, 3)
        self.charlie = WithDefaults()
        self.delta = {'echo': 'E', 'nest': {0: 'a', 1: 'b'}}
        self.echo = ['alex', 'bethany']
        self.fox = ('carry', 'divide')
        self.gamma = 'myexcitingstring'