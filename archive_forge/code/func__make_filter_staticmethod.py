import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def _make_filter_staticmethod(self, ignore_classes=AssertionError):

    class ExceptionIgnorer(object):

        @excutils.exception_filter
        @staticmethod
        def ignore_exceptions(ex):
            """Ignore some exceptions S."""
            return isinstance(ex, ignore_classes)
    return ExceptionIgnorer.ignore_exceptions