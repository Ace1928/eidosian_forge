from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class WithHelpArg(object):
    """Test class for testing when class has a help= arg."""

    def __init__(self, help=True):
        self.has_help = help
        self.dictionary = {'__help': 'help in a dict'}