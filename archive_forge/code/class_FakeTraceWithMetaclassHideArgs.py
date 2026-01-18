import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceWithMetaclassHideArgs(FakeTraceWithMetaclassBase):
    __trace_args__ = {'name': 'a', 'info': {'b': 20}, 'hide_args': True}

    def method5(self, k, l):
        return k + l