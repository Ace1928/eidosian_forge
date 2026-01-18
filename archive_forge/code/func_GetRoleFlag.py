from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam.byoid_utilities import cred_config
from googlecloudsdk.command_lib.util.args import common_args
def GetRoleFlag(verb):
    return base.Argument('role', metavar='ROLE_ID', help='ID of the role to {0}. Curated roles example: roles/viewer. Custom roles example: CustomRole. For custom roles, you must also specify the `--organization` or `--project` flag.'.format(verb))