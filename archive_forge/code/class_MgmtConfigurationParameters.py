import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
class MgmtConfigurationParameters(configurations.ConfigurationParameters):

    def create(self, version, name, restart_required, data_type, max_size=None, min_size=None):
        """Mgmt call to create a new configuration parameter."""
        body = {'configuration-parameter': {'name': name, 'restart_required': int(restart_required), 'data_type': data_type}}
        if max_size is not None:
            body['configuration-parameter']['max_size'] = max_size
        if min_size is not None:
            body['configuration-parameter']['min_size'] = min_size
        url = '/mgmt/datastores/versions/%s/parameters' % version
        resp, body = self.api.client.post(url, body=body)
        common.check_for_exceptions(resp, body, url)

    def list_all_parameter_by_version(self, version):
        """List all configuration parameters deleted or not."""
        return self._list('/mgmt/datastores/versions/%s/parameters' % version, 'configuration-parameters')

    def get_any_parameter_by_version(self, version, key):
        """Get any configuration parameter deleted or not."""
        return self._get('/mgmt/datastores/versions/%s/parameters/%s' % (version, key))

    def modify(self, version, name, restart_required, data_type, max_size=None, min_size=None):
        """Mgmt call to modify an existing configuration parameter."""
        body = {'configuration-parameter': {'name': name, 'restart_required': int(restart_required), 'data_type': data_type}}
        if max_size is not None:
            body['configuration-parameter']['max_size'] = max_size
        if min_size is not None:
            body['configuration-parameter']['min_size'] = min_size
        output = {'version': version, 'parameter_name': name}
        url = '/mgmt/datastores/versions/%(version)s/parameters/%(parameter_name)s' % output
        resp, body = self.api.client.put(url, body=body)
        common.check_for_exceptions(resp, body, url)

    def delete(self, version, name):
        """Mgmt call to delete a configuration parameter."""
        output = {'version_id': version, 'parameter_name': name}
        url = '/mgmt/datastores/versions/%(version_id)s/parameters/%(parameter_name)s' % output
        resp, body = self.api.client.delete(url)
        common.check_for_exceptions(resp, body, url)