from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class YamlConfigObjectError(core_exceptions.Error):
    """Raised when an invalid Operation is invoked on YamlConfigObject."""