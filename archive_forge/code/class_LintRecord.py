from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
class LintRecord(object):
    """Records an instance of lint at a particular location
  """

    def __init__(self, spec, location, msg):
        self.spec = spec
        self.location = location
        self.msg = msg

    def __repr__(self):
        if self.location is None:
            return ' [{:s}] {:s}'.format(self.spec.idstr, self.msg)
        if isinstance(self.location, tuple):
            return '{:s}: [{:s}] {:s}'.format(','.join(('{:02d}'.format(val) for val in self.location[:2])), self.spec.idstr, self.msg)
        raise ValueError('Unexpected type {} for location'.format(type(self.location)))