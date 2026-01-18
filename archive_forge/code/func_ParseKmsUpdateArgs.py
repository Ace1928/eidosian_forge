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
def ParseKmsUpdateArgs(args):
    """Parses KMS key value."""
    location_id = args.location if args.location else None
    full_kms_key_name = None
    parse_result = ParseFullKmsKeyName(args.kms_key_name)
    if parse_result is not None:
        location_id = parse_result[1]
        full_kms_key_name = args.kms_key_name
    elif args.kms_key_name and args.kms_keyring and args.location:
        full_kms_key_name = 'projects/{kms_project_id}/locations/{location_id}/keyRings/{kms_keyring}/cryptoKeys/{kms_key_name}'.format(kms_project_id=args.kms_project if args.kms_project else _PROJECT(), location_id=location_id, kms_keyring=args.kms_keyring, kms_key_name=args.kms_key_name)
    return (_PROJECT(), full_kms_key_name, location_id)