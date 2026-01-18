import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class DummyTask(object):

    def __init__(self, num_steps=3, delays=None):
        self.num_steps = num_steps
        if delays is not None:
            self.delays = iter(delays)
        else:
            self.delays = itertools.repeat(None)

    def __call__(self, *args, **kwargs):
        for i in range(1, self.num_steps + 1):
            self.do_step(i, *args, **kwargs)
            yield next(self.delays)

    def do_step(self, step_num, *args, **kwargs):
        pass