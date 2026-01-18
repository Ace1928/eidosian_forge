from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def AddFileArg(parser):
    """Adds a FILE positional arg."""
    parser.add_argument('FILE', type=arg_parsers.YAMLFileContents(), help='The absolute path to the YAML file with an application definition to update or deploy.')