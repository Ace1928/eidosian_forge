from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
class InvalidResourcePathError(Error):
    """Raised when a resources.yaml is not found by the given resource_path."""