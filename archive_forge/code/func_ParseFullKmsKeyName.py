from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def ParseFullKmsKeyName(kms_key_name):
    """Parses and retrieves the segments of a full KMS key name."""
    if not kms_key_name:
        return None
    match = re.match('projects\\/(?P<project>.*)\\/locations\\/(?P<location>.*)\\/keyRings\\/(?P<keyring>.*)\\/cryptoKeys\\/(?P<key>.*)', kms_key_name)
    if match:
        return [match.group('project'), match.group('location'), match.group('keyring'), match.group('key')]
    return None