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
def GenerateAttributeBindingPath(args):
    """Generate Data Attribute Binding Path."""
    if args.IsSpecified('path_file_name'):
        return GenerateAttributeBindingPathFromFile(args.path_file_name, args.path_file_format)
    else:
        return GenerateAttributeBindingPathFromParam(args)