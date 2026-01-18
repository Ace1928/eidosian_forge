import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
class Fail2(excutils.CausedByException):
    pass