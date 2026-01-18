from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _cluster_users(self, args: parser_extensions.Namespace):
    """Constructs repeated proto message ClusterUser."""
    cluster_user_messages = []
    admin_users = getattr(args, 'admin_users', None)
    if admin_users:
        return [messages.ClusterUser(username=admin_user) for admin_user in admin_users]
    gcloud_config_core_account = properties.VALUES.core.account.Get()
    if gcloud_config_core_account:
        default_admin_user_message = messages.ClusterUser(username=gcloud_config_core_account)
        cluster_user_messages.append(default_admin_user_message)
        return cluster_user_messages
    return None