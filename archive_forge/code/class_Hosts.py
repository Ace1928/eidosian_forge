from troveclient import base
from troveclient import common
class Hosts(base.ManagerWithFind):
    """Manage :class:`Host` resources."""
    resource_class = Host

    def _list(self, url, response_key):
        resp, body = self.api.client.get(url)
        if not body:
            raise Exception('Call to ' + url + ' did not return a body.')
        return [self.resource_class(self, res) for res in body[response_key]]

    def _action(self, host_id, body):
        """Perform a host "action" -- update."""
        url = '/mgmt/hosts/%s/instances/action' % host_id
        resp, body = self.api.client.post(url, body=body)
        common.check_for_exceptions(resp, body, url)

    def update_all(self, host_id):
        """Update all instances on a host."""
        body = {'update': ''}
        self._action(host_id, body)

    def index(self):
        """Get a list of all hosts.

        :rtype: list of :class:`Hosts`.
        """
        return self._list('/mgmt/hosts', 'hosts')

    def get(self, host):
        """Get a specific host.

        :rtype: :class:`host`
        """
        return self._get('/mgmt/hosts/%s' % self._get_host_name(host), 'host')

    @staticmethod
    def _get_host_name(host):
        try:
            if host.name:
                return host.name
        except AttributeError:
            return host

    def list(self):
        pass