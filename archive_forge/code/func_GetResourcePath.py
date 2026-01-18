from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware.sddc import util
from googlecloudsdk.command_lib.vmware.sddc import flags
import six.moves.urllib.parse
def GetResourcePath(self, resource, resource_path, encoded_cluster_groups_id=False):
    result = six.text_type(resource_path)
    if '/' not in resource.clusterGroupsId:
        return result
    cluster_groups_id = resource.clusterGroupsId.split('/').pop()
    cluster_groups_id_path = six.text_type(resource.clusterGroupsId)
    if encoded_cluster_groups_id:
        cluster_groups_id_path = six.moves.urllib.parse.quote(cluster_groups_id_path, safe='')
    return result.replace(cluster_groups_id_path, cluster_groups_id)