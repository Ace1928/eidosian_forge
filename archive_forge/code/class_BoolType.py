import functools
from unittest import mock
import uuid
from keystoneauth1 import loading
from keystoneauth1.loading import base
from keystoneauth1 import plugin
from keystoneauth1.tests.unit import utils
class BoolType(object):

    def __eq__(self, other):
        """Define equiality for many bool types."""
        return type(self) is type(other)

    def __ne__(self, other):
        """Define inequiality for many bool types."""
        return not self.__eq__(other)

    def __call__(self, value):
        return str(value).lower() in ('1', 'true', 't', 'yes', 'y')