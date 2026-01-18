import re
import unittest
from wsme import exc
from wsme import types
class buffer:

    def read(self):
        return 'from-file'