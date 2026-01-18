from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
def _GRIsAreEnabled():
    """Returns True if GRIs are enabled."""
    return properties.VALUES.core.enable_gri.GetBool() or properties.VALUES.core.resource_completion_style.Get() == 'gri'