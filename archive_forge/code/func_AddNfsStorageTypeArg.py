from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNfsStorageTypeArg(parser):
    """Adds storage type argument for NFS."""
    parser.add_argument('--storage-type', choices={'SSD': 'The storage type of the underlying volume will be SSD', 'HDD': 'The storage type of the underlying volume will be HDD'}, required=True, type=arg_utils.ChoiceToEnumName, help='Specifies the storage type of the underlying volume which will be created for the NFS share.')