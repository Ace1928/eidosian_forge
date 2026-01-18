from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class FakeRepo(object):

    def __init__(self, result=None):
        self.args = None
        self.kwargs = None
        self.result = result

    def fake_method(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.result
    get = fake_method
    list = fake_method
    add = fake_method
    save = fake_method
    remove = fake_method
    set_property_atomic = fake_method
    delete_property_atomic = fake_method