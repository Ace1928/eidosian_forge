import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
class MgmtClusters(base.ManagerWithFind):
    """Manage :class:`Cluster` resources."""
    resource_class = clusters.Cluster

    def list(self):
        pass

    def show(self, cluster):
        """Get details of one cluster."""
        return self._get('/mgmt/clusters/%s' % base.getid(cluster), 'cluster')

    def index(self, deleted=None, limit=None, marker=None):
        """Show an overview of all local clusters.

        Optionally, filter by deleted status.

        :rtype: list of :class:`Cluster`.
        """
        form = ''
        if deleted is not None:
            if deleted:
                form = '?deleted=true'
            else:
                form = '?deleted=false'
        url = '/mgmt/clusters%s' % form
        return self._paginated(url, 'clusters', limit, marker)

    def _action(self, cluster_id, body):
        """Perform a cluster action, e.g. reset-task."""
        url = '/mgmt/clusters/%s/action' % cluster_id
        resp, body = self.api.client.post(url, body=body)
        common.check_for_exceptions(resp, body, url)

    def reset_task(self, cluster_id):
        """Reset the current cluster task to NONE."""
        body = {'reset-task': {}}
        self._action(cluster_id, body)