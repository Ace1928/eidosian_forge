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
class ResourcedToStreamDecorator(ExtendedToStreamDecorator):
    """Report ``testresources``-related activity to StreamResult objects.

    Implement the resource lifecycle TestResult protocol extension supported
    by the ``testresources.TestResourceManager`` class. At each stage of a
    resource's lifecycle, a stream event with relevant details will be
    emitted.

    Each stream event will have its test_id field set to the resource manager's
    identifier (see ``testresources.TestResourceManager.id()``) plus the method
    being executed (either 'make' or 'clean').

    The test_status will be either 'inprogress' or 'success'.

    The runnable flag will be set to False.
    """

    def startMakeResource(self, resource):
        self._convertResourceLifecycle(resource, 'make', 'start')

    def stopMakeResource(self, resource):
        self._convertResourceLifecycle(resource, 'make', 'stop')

    def startCleanResource(self, resource):
        self._convertResourceLifecycle(resource, 'clean', 'start')

    def stopCleanResource(self, resource):
        self._convertResourceLifecycle(resource, 'clean', 'stop')

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