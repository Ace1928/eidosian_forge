from oslo_log import log as logging
from heat.common.i18n import _
from heat.engine import resource
from heat.engine import support
class BaseSenlinResource(resource.Resource):
    """A base class for Senlin resources."""
    support_status = support.SupportStatus(version='22.0.0', status=support.DEPRECATED, message=_('Senlin project was marked inactive'), previous_status=support.SupportStatus(version='6.0.0'))
    default_client_name = 'senlin'

    def _show_resource(self):
        method_name = 'get_' + self.entity
        try:
            client_method = getattr(self.client(), method_name)
            res_info = client_method(self.resource_id)
            return res_info.to_dict()
        except AttributeError as ex:
            LOG.warning('No method to get the resource: %s', ex)

    def _resolve_attribute(self, name):
        if self.resource_id is None:
            return
        res_info = self._show_resource()
        return res_info.get(name)