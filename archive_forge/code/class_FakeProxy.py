from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class FakeProxy(object):

    def __init__(self, base, *args, **kwargs):
        self.base = base
        self.args = args
        self.kwargs = kwargs