from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
class StrException(Exception):

    def __init__(self, value):
        Exception.__init__(self)
        self.value = value

    def __str__(self):
        return self.value