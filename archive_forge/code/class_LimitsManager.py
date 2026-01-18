from cinderclient import base
from cinderclient import utils
class LimitsManager(base.Manager):
    """Manager object used to interact with limits resource."""
    resource_class = Limits

    def get(self, tenant_id=None):
        """Get a specific extension.

        :rtype: :class:`Limits`
        """
        opts = {}
        if tenant_id:
            opts['tenant_id'] = tenant_id
        query_string = utils.build_query_param(opts)
        return self._get('/limits%s' % query_string, 'limits')