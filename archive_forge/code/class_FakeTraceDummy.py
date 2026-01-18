import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceDummy(FakeTraceWithMetaclassBase):

    def method4(self, j):
        return j