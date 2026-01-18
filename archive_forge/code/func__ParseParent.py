import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def _ParseParent(parent: str) -> str:
    """Extracts parent name from a string of the form {organizations|projects|folders}/<id>."""
    if parent.startswith('organizations/'):
        return resources.REGISTRY.Parse(parent, collection='cloudresourcemanager.organizations')
    elif parent.startswith('folders/'):
        return folders.FoldersRegistry().Parse(parent, collection='cloudresourcemanager.folders')
    elif parent.startswith('projects/'):
        return resources.REGISTRY.Parse(parent, collection='cloudresourcemanager.projects')
    else:
        raise errors.InvalidParentError(parent)