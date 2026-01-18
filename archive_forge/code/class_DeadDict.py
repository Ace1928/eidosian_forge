from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
class DeadDict(dict):
    _asdict = None