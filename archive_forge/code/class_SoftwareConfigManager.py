from urllib import parse
from heatclient.common import base
from heatclient.common import utils
class SoftwareConfigManager(base.BaseManager):
    resource_class = SoftwareConfig

    def list(self, **kwargs):
        """Get a list of software configs.

        :rtype: list of :class:`SoftwareConfig`
        """
        qparams = {}
        for opt, val in kwargs.items():
            if val:
                qparams[opt] = val
        if qparams:
            new_qparams = sorted(qparams.items(), key=lambda x: x[0])
            query_string = '?%s' % parse.urlencode(new_qparams)
        else:
            query_string = ''
        url = '/software_configs%s' % query_string
        return self._list(url, 'software_configs')

    def get(self, config_id):
        """Get the details for a specific software config.

        :param config_id: ID of the software config
        """
        resp = self.client.get('/software_configs/%s' % config_id)
        body = utils.get_response_body(resp)
        return SoftwareConfig(self, body.get('software_config'))

    def create(self, **kwargs):
        """Create a software config."""
        resp = self.client.post('/software_configs', data=kwargs)
        body = utils.get_response_body(resp)
        return SoftwareConfig(self, body.get('software_config'))

    def delete(self, config_id):
        """Delete a software config."""
        self._delete('/software_configs/%s' % config_id)