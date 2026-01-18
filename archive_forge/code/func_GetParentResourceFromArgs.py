from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetParentResourceFromArgs(args):
    """Returns the resource from the user-specified arguments.

  A resource has the following syntax:
  [organizations|folders|projects]/{resource_id}.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    resource_id = args.organization or args.folder or args.project
    if args.organization:
        resource_type = ORGANIZATION
    elif args.folder:
        resource_type = FOLDER
    else:
        resource_type = PROJECT
    return '{}/{}'.format(resource_type + 's', resource_id)