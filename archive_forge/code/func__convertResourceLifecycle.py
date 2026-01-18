import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _convertResourceLifecycle(self, resource, method, phase):
    """Convert a resource lifecycle report to a stream event."""
    if hasattr(resource, 'id'):
        resource_id = resource.id()
    else:
        resource_id = '{}.{}'.format(resource.__class__.__module__, resource.__class__.__name__)
    test_id = f'{resource_id}.{method}'
    if phase == 'start':
        test_status = 'inprogress'
    else:
        test_status = 'success'
    self.status(test_id=test_id, test_status=test_status, runnable=False, timestamp=self._now())