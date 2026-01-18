from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class FakeImageFactory(object):

    def __init__(self, result=None):
        self.result = None
        self.kwargs = None

    def new_image(self, **kwargs):
        self.kwargs = kwargs
        return self.result