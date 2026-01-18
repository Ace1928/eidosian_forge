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
def GenerateResourceAccessSpec(args):
    """Generate Resource Access Spec From Arguments."""
    module = dataplex_api.GetMessageModule()
    resource_access_spec = module.GoogleCloudDataplexV1ResourceAccessSpec(owners=args.resource_owners, readers=args.resource_readers, writers=args.resource_writers)
    return resource_access_spec