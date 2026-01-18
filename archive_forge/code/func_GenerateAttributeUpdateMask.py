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
def GenerateAttributeUpdateMask(args):
    """Create Update Mask for DataTaxonomy."""
    update_mask = []
    if args.IsSpecified('description'):
        update_mask.append('description')
    if args.IsSpecified('display_name'):
        update_mask.append('displayName')
    if args.IsSpecified('labels'):
        update_mask.append('labels')
    if args.IsSpecified('parent'):
        update_mask.append('parentId')
    if args.IsSpecified('resource_readers'):
        update_mask.append('resourceAccessSpec.readers')
    if args.IsSpecified('resource_writers'):
        update_mask.append('resourceAccessSpec.writers')
    if args.IsSpecified('resource_owners'):
        update_mask.append('resourceAccessSpec.owners')
    if args.IsSpecified('data_readers'):
        update_mask.append('dataAccessSpec.readers')
    return update_mask