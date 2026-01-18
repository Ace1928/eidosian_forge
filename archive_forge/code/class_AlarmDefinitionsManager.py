from monascaclient.common import monasca_manager
class AlarmDefinitionsManager(monasca_manager.MonascaManager):
    base_url = '/alarm-definitions'

    def create(self, **kwargs):
        """Create an alarm definition."""
        resp = self.client.create(url=self.base_url, json=kwargs)
        return resp

    def get(self, **kwargs):
        """Get the details for a specific alarm definition."""
        url = '%s/%s' % (self.base_url, kwargs['alarm_id'])
        resp = self.client.list(path=url)
        return resp

    def list(self, **kwargs):
        """Get a list of alarm definitions."""
        return self._list('', 'dimensions', **kwargs)

    def delete(self, **kwargs):
        """Delete a specific alarm definition."""
        url_str = self.base_url + '/%s' % kwargs['alarm_id']
        resp = self.client.delete(url_str)
        return resp

    def update(self, **kwargs):
        """Update a specific alarm definition."""
        url_str = self.base_url + '/%s' % kwargs['alarm_id']
        del kwargs['alarm_id']
        resp = self.client.create(url=url_str, method='PUT', json=kwargs)
        return resp

    def patch(self, **kwargs):
        """Patch a specific alarm definition."""
        url_str = self.base_url + '/%s' % kwargs['alarm_id']
        del kwargs['alarm_id']
        resp = self.client.create(url=url_str, method='PATCH', json=kwargs)
        return resp