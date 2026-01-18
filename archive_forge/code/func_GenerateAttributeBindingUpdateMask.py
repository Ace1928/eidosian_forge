from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def GenerateAttributeBindingUpdateMask(args):
    """Create Update Mask for DataAttributeBinding."""
    update_mask = []
    if args.IsSpecified('description'):
        update_mask.append('description')
    if args.IsSpecified('display_name'):
        update_mask.append('displayName')
    if args.IsSpecified('labels'):
        update_mask.append('labels')
    if args.IsSpecified('resource_attributes'):
        update_mask.append('attributes')
    if args.IsSpecified('paths'):
        update_mask.append('paths')
    if args.IsSpecified('path_file_name'):
        update_mask.append('paths')
    if args.IsSpecified('etag'):
        update_mask.append('etag')
    return update_mask