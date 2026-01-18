from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def SerializeToJsonOrYaml(authorized_view, pre_encoded, serialized_format='json'):
    """Serializes a authorized view protobuf to either JSON or YAML."""
    authorized_view_dict = encoding.MessageToDict(authorized_view)
    if not pre_encoded:
        authorized_view_dict = Base64DecodingYamlAuthorizedViewDefinition(authorized_view_dict)
    if serialized_format == 'json':
        return six.text_type(json.dumps(authorized_view_dict, indent=2))
    if serialized_format == 'yaml':
        return six.text_type(yaml.dump(authorized_view_dict))