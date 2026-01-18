import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace_cls('rpc', info={'a': 10})
class FakeTraceClassWithInfo(FakeTracedCls):
    pass