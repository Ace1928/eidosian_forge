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
def _GetParentResourceFromArgs(args):
    if args.organization:
        return resources.REGISTRY.Parse(args.organization, collection='cloudresourcemanager.organizations')
    elif args.folder:
        return folders.FoldersRegistry().Parse(args.folder, collection='cloudresourcemanager.folders')
    else:
        return resources.REGISTRY.Parse(args.project or properties.VALUES.core.project.Get(required=True), collection='cloudresourcemanager.projects')