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
def GenerateDataAttributeForUpdateRequest(data_attribute_ref, args):
    """Generate attributes for Update Data Attribute Requests."""
    module = dataplex_api.GetMessageModule()
    request = module.GoogleCloudDataplexV1DataAttribute(description=args.description, displayName=args.display_name, parentId=ResolveParentId(data_attribute_ref, args), resourceAccessSpec=GenerateResourceAccessSpec(args), dataAccessSpec=GenerateDataAccessSpec(args), etag=args.etag, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1DataAttribute, args))
    return request