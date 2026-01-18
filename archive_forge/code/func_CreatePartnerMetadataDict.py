from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
def CreatePartnerMetadataDict(args):
    """create partner metadata from the given args.

  Args:
    args: args containing partner-metadata or partner-metadata-from-file flags

  Returns:
    python dict contains partner metadata from given args.
  """
    return _CreatePartnerMetadataDict(args.partner_metadata, args.partner_metadata_from_file)