from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def FeatureEnabledInMetadata(instance, project, key_name, instance_override=None):
    """Return True if the feature associated with the supplied key is enabled.

  If the key is set to 'true' in instance metadata, will return True.
  If the key is set to 'false' in instance metadata, will return False.
  If key is not set in instance metadata, will return the value in project
  metadata unless instance_override is not None (set to True or False).

  Args:
    instance: The current instance object.
    project: The current project object.
    key_name: The name of metadata key to check. e.g. 'oslogin-enable'.
    instance_override: The value of the instance metadata key. Used if the
      instance object cannot be passed in. None if not set.

  Returns:
    bool, True if the feature associated with the supplied key is enabled
      in instance/project metadata.
  """
    feature_enabled = None
    if instance is not None:
        feature_enabled = MetadataHasEnable(instance.metadata, key_name)
    elif instance_override is not None:
        feature_enabled = instance_override
    if feature_enabled is None:
        project_metadata = project.commonInstanceMetadata
        feature_enabled = MetadataHasEnable(project_metadata, key_name)
    return feature_enabled