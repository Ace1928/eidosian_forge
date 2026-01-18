from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
class PositionalSpec(tuple):
    """
  Encapsulates the parse specification for a positional argument group.
  NOTE(josh): this is a named tuple with default arguments and some init
  processing. If it wasn't for the init processing, we could just do:

  PositionalSpec = collections.namedtuple(
    "PositionalSpec", ["nargs", ...])
  PositionalSpec.__new__.__defaults__ = (False, None, None, False)

  But we don't want to self.tags and self.flags to point to a mutable global
  variable...
  """

    def __new__(cls, nargs, sortable=False, tags=None, flags=None, legacy=False, max_pargs_hwrap=None, always_wrap=None):
        if not tags:
            tags = []
        if not flags:
            flags = []
        return tuple.__new__(cls, (nargs, sortable, tags, flags, legacy, max_pargs_hwrap, always_wrap))
    nargs = property(_itemgetter(0))
    npargs = property(_itemgetter(0))
    sortable = property(_itemgetter(1))
    tags = property(_itemgetter(2))
    flags = property(_itemgetter(3))
    legacy = property(_itemgetter(4))
    max_pargs_hwrap = property(_itemgetter(5))
    always_wrap = property(_itemgetter(6))

    def replace(self, **kwargs):
        selfdict = {'nargs': self.nargs, 'sortable': self.sortable, 'tags': list(self.tags), 'flags': list(self.flags), 'legacy': self.legacy, 'max_pargs_hwrap': self.max_pargs_hwrap, 'always_wrap': self.always_wrap}
        selfdict.update(kwargs)
        return PositionalSpec(**selfdict)