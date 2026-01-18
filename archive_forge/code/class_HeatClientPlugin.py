from oslo_config import cfg
from heatclient import client as hc
from heatclient import exc
from heat.engine.clients import client_plugin
class HeatClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = exc
    service_types = [ORCHESTRATION, CLOUDFORMATION] = ['orchestration', 'cloudformation']

    def _create(self):
        endpoint = self.get_heat_url()
        args = {}
        if self._get_client_option(CLIENT_NAME, 'url'):
            args['username'] = self.context.username
            args['password'] = self.context.password
        args['connect_retries'] = cfg.CONF.client_retry_limit
        return hc.Client('1', endpoint_override=endpoint, session=self.context.keystone_session, **args)

    def is_not_found(self, ex):
        return isinstance(ex, exc.HTTPNotFound)

    def is_over_limit(self, ex):
        return isinstance(ex, exc.HTTPOverLimit)

    def is_conflict(self, ex):
        return isinstance(ex, exc.HTTPConflict)

    def get_heat_url(self):
        heat_url = self._get_client_option(CLIENT_NAME, 'url')
        if heat_url:
            tenant_id = self.context.tenant_id
            heat_url = heat_url % {'tenant_id': tenant_id}
        else:
            endpoint_type = self._get_client_option(CLIENT_NAME, 'endpoint_type')
            heat_url = self.url_for(service_type=self.ORCHESTRATION, endpoint_type=endpoint_type)
        return heat_url

    def get_heat_cfn_url(self):
        endpoint_type = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        heat_cfn_url = self.url_for(service_type=self.CLOUDFORMATION, endpoint_type=endpoint_type)
        return heat_cfn_url

    def get_cfn_metadata_server_url(self):
        config_url = cfg.CONF.heat_metadata_server_url
        if config_url is None:
            config_url = self.get_heat_cfn_url()
        if '/v1' not in config_url:
            config_url += '/v1'
        if config_url and config_url[-1] != '/':
            config_url += '/'
        return config_url

    def get_insecure_option(self):
        return self._get_client_option(CLIENT_NAME, 'insecure')