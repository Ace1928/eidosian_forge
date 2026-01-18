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
def GenerateAttributeBindingPathFromParam(args):
    """Create Path from specified path parameter."""
    module = dataplex_api.GetMessageModule()
    attribute_binding_path = []
    if args.paths is not None:
        for path in args.paths:
            attribute_binding_path.append(module.GoogleCloudDataplexV1DataAttributeBindingPath(name=path.get('name'), attributes=path.get('attributes')))
    return attribute_binding_path