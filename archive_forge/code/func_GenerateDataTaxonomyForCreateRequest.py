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
def GenerateDataTaxonomyForCreateRequest(args):
    """Create Data Taxonomy Requests."""
    module = dataplex_api.GetMessageModule()
    request = module.GoogleCloudDataplexV1DataTaxonomy(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1DataTaxonomy, args))
    return request