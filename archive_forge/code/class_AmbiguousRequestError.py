from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class AmbiguousRequestError(exceptions.Error):
    """Raised when the user makes a request for an ambiguously defined resource.

  Sometimes arguments are optional in the general case because their correct
  values can generally be inferred, but required for cases when that inferrence
  isn't possible. This error covers that scenario.
  """
    pass