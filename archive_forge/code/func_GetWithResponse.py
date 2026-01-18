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
def GetWithResponse(self, response_func):
    for op in self._operation_refs:
        if response_func(self._responses.get(op)):
            yield op