from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import enum
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
class ResourceFileTypeError(exceptions.Error):
    """Indicates a resource file was of an unsupported type."""