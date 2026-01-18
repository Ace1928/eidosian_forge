from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def _BreakpointDescription(self, restrict_to_type):
    if not restrict_to_type:
        return 'breakpoint'
    elif restrict_to_type == self.SNAPSHOT_TYPE:
        return 'snapshot'
    else:
        return 'logpoint'