import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace_cls('rpc', trace_static_methods=True)
class FakeTraceStaticMethod(FakeTraceStaticMethodBase):
    pass