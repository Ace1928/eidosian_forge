from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import textwrap
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core.util import files
import six
class ContainerNotFoundException(core_exceptions.Error):
    """Docker container is not found."""