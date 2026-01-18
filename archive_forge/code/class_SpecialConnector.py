from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
class SpecialConnector(enum.Enum):
    """Special cases for ReachableSecret._connector."""
    PATH_OR_ENV = 0