from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
import six
from six.moves import zip
class OperationBatch(object):
    """Wrapper class for a set of batched operations."""

    def __init__(self, operation_refs):
        self._operation_refs = operation_refs or []
        self._responses = {}

    def SetResponse(self, operation_ref, response):
        self._responses[operation_ref] = response

    def GetResponse(self, operation_ref):
        return self._responses.get(operation_ref)

    def GetWithResponse(self, response_func):
        for op in self._operation_refs:
            if response_func(self._responses.get(op)):
                yield op

    def __iter__(self):
        return iter(self._operation_refs)

    def __str__(self):
        return '[{0}]'.format(', '.join((six.text_type(r) for r in self._operation_refs)))