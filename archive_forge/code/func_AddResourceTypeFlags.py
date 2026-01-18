from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddResourceTypeFlags(parser):
    """Add resource-type flag to parser."""
    group = parser.add_group(mutex=True, required=False, help='`RESOURCE TYPE FILTERS` - specify resource types to export.')
    group.add_argument('--resource-types', type=arg_parsers.ArgList(), metavar='RESOURCE_TYPE', help='List of Config Connector KRM Kinds to export.\n  For a full list of supported resource types for a given parent scope run:\n\n  $ {parent_command} list-resource-types --[project|organization|folder]=<PARENT>\n  ')
    group.add_argument('--resource-types-file', type=arg_parsers.FileContents(), metavar='RESOURCE_TYPE_FILE', help="A comma (',') or newline ('\\n') separated file containing the list of\n      Config Connector KRM Kinds to export.\n  For a full list of supported resource types for a given parent scope run:\n\n  $ {parent_command} list-resource-types --[project|organization|folder]=<PARENT>\n  ")