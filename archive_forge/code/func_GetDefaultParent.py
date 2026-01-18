from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetDefaultParent():
    """Converts user input to one of: organization, project, or folder."""
    organization_resource_pattern = re.compile('organizations/[0-9]+$')
    id_pattern = re.compile('[0-9]+')
    parent = properties.VALUES.scc.parent.Get()
    if id_pattern.match(parent):
        parent = 'organizations/' + parent
    if not (organization_resource_pattern.match(parent) or parent.startswith('projects/') or parent.startswith('folders/')):
        raise errors.InvalidSCCInputError('Parent must match either [0-9]+, organizations/[0-9]+, projects/.*\n      or folders/.*.')
    return parent