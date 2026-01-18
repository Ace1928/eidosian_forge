from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import base64
import collections
import json
import os
import os.path
import re
import uuid
from apitools.base.py import encoding_helper
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import yaml_parsing as app_engine_yaml_parsing
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import service as k8s_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import secrets
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def _AddSecretVolumeByName(volumes, secret_name, volume_name, items=None):
    """Add a secret volume to a list of volumes.

  Args:
    volumes: (list[dict]) List of volume specifications.
    secret_name: (str) Name of the secret.
    volume_name: (str) Name of the volume to add.
    items: (list[_SecretPath]) Optional list of SecretPaths to map on the
      filesystem.
  """
    if any((volume['name'] == volume_name for volume in volumes)):
        return
    items_lst = [RUN_MESSAGES_MODULE.KeyToPath(key=secpath.key, path=secpath.path) for secpath in items or []]
    volumes.append(encoding_helper.MessageToDict(RUN_MESSAGES_MODULE.Volume(name=volume_name, secret=RUN_MESSAGES_MODULE.SecretVolumeSource(secretName=secret_name, items=items_lst))))