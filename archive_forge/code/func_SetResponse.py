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
def SetResponse(self, operation_ref, response):
    self._responses[operation_ref] = response