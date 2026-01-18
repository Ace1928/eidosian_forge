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
def _CreatePartnerMetadataDict(partner_metadata, partner_metadata_from_file=None):
    """create partner metadata from the given args.

  Args:
    partner_metadata: partner metadata dictionary.
    partner_metadata_from_file: partner metadata file content.

  Returns:
    python dict contains partner metadata from given args.
  """
    partner_metadata_file = {}
    if partner_metadata_from_file:
        partner_metadata_file = json.loads(partner_metadata_from_file)
    partner_metadata_dict = {}
    for key in partner_metadata_file.keys():
        if 'entries' in partner_metadata_file[key]:
            partner_metadata_dict[key] = partner_metadata_file[key]
        else:
            partner_metadata_dict[key] = {'entries': partner_metadata_file[key]}
    for key, value in partner_metadata.items():
        namespace, entry = key.split('/')
        if namespace not in partner_metadata_dict:
            partner_metadata_dict[namespace] = {'entries': {}}
        partner_metadata_dict[namespace]['entries'][entry] = json.loads(value)
    return partner_metadata_dict