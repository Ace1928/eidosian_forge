from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def _MetadataHasGuestAttributesEnabled(metadata):
    """Returns true if the metadata has 'enable-guest-attributes' set to 'true'.

  Args:
    metadata: Instance or Project metadata

  Returns:
    True if Enabled, False if Disabled, None if key is not present.
  """
    if not (metadata and metadata.items):
        return None
    matching_values = [item.value for item in metadata.items if item.key == GUEST_ATTRIBUTES_METADATA_KEY]
    if not matching_values:
        return None
    return matching_values[0].lower() == 'true'