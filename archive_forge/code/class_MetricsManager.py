from monascaclient.common import monasca_manager
class MetricsManager(monasca_manager.MonascaManager):
    base_url = '/metrics'

    def create(self, **kwargs):
        """Create a metric."""
        url_str = self.base_url
        if 'tenant_id' in kwargs:
            url_str = url_str + '?tenant_id=%s' % kwargs['tenant_id']
            del kwargs['tenant_id']
        data = kwargs['jsonbody'] if 'jsonbody' in kwargs else kwargs
        body = self.client.create(url=url_str, json=data)
        return body

    def list(self, **kwargs):
        """Get a list of metrics."""
        return self._list('', 'dimensions', **kwargs)

    def list_names(self, **kwargs):
        """Get a list of metric names."""
        return self._list('/names', 'dimensions', **kwargs)

    def list_measurements(self, **kwargs):
        """Get a list of measurements based on metric definition filters."""
        return self._list('/measurements', 'dimensions', **kwargs)

    def list_statistics(self, **kwargs):
        """Get a list of measurement statistics based on metric def filters."""
        return self._list('/statistics', 'dimensions', **kwargs)

    def list_dimension_names(self, **kwargs):
        """Get a list of metric dimension names."""
        return self._list('/dimensions/names', **kwargs)

    def list_dimension_values(self, **kwargs):
        """Get a list of metric dimension values."""
        return self._list('/dimensions/names/values', **kwargs)