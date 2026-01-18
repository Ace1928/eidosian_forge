import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceWithMetaclassPrivate(FakeTraceWithMetaclassBase):
    __trace_args__ = {'name': 'rpc', 'trace_private': True}

    def _new_private_method(self, m):
        return 2 * m