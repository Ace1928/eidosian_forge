from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _refresh_cluster(self, cluster_id):
    try:
        cluster = self.client().clusters.get(cluster_id)
        return cluster
    except Exception as exc:
        if self.client_plugin().is_over_limit(exc):
            LOG.warning('Stack %(name)s (%(id)s) received an OverLimit response during clusters.get(): %(exception)s', {'name': self.stack.name, 'id': self.stack.id, 'exception': exc})
            return None
        else:
            raise