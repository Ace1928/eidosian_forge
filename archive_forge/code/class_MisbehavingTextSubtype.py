from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
class MisbehavingTextSubtype(text_type):

    def __str__(self):
        return 'FAIL!'