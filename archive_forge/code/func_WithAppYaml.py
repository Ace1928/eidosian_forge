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
def WithAppYaml(self, yaml_path):
    """Overrides settings with app.yaml and returns a new Settings object.

    Args:
      yaml_path: Filename to read.

    Returns:
      New Settings object.

    Raises:
      ParseError: Input does not look like an app.yaml
    """
    try:
        service_config = app_engine_yaml_parsing.ServiceYamlInfo.FromFile(yaml_path)
    except yaml.Error:
        raise _ParseError()
    builder_url = _GaeBuilderPackagePath(service_config.parsed.runtime)
    builder = builders.BuildpackBuilder(builder=builder_url, trust=True, devmode=False)
    return self.replace(builder=builder)