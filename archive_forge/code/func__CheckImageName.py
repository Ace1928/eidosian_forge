from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
def _CheckImageName(image_name):
    """Checks for a valid GCE image name."""
    name_message = 'Name must start with a lowercase letter followed by up to 63 lowercase letters, numbers, or hyphens, and cannot end with a hyphen.'
    name_ok = True
    valid_chars = string.digits + string.ascii_lowercase + '-'
    if len(image_name) > 64:
        name_ok = False
    elif image_name[0] not in string.ascii_lowercase:
        name_ok = False
    elif not all((char in valid_chars for char in image_name)):
        name_ok = False
    elif image_name[-1] == '-':
        name_ok = False
    if not name_ok:
        raise exceptions.InvalidArgumentException('IMAGE_NAME', name_message)