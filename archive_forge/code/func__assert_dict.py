import logging
import os
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import initializer
from osprofiler import opts
from osprofiler import profiler
from osprofiler.tests import test
def _assert_dict(self, info, **kwargs):
    for key in kwargs:
        self.assertEqual(kwargs[key], info[key])