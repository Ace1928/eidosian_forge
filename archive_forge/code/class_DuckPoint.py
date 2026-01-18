from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
class DuckPoint(object):

    def __init__(self, *args):
        self.point = Point(*args)

    def _asdict(self):
        return self.point._asdict()