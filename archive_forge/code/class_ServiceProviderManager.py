from keystoneclient import base
class ServiceProviderManager(base.CrudManager):
    """Manager class for manipulating Service Providers."""
    resource_class = ServiceProvider
    collection_key = 'service_providers'
    key = 'service_provider'
    base_url = 'OS-FEDERATION'

    def _build_url_and_put(self, **kwargs):
        url = self.build_url(dict_args_in_out=kwargs)
        body = {self.key: kwargs}
        return self._update(url, body=body, response_key=self.key, method='PUT')

    def create(self, id, **kwargs):
        """Create Service Provider object.

        Utilize Keystone URI:
        ``PUT /OS-FEDERATION/service_providers/{id}``

        :param id: unique id of the service provider.

        """
        return self._build_url_and_put(service_provider_id=id, **kwargs)

    def get(self, service_provider):
        """Fetch Service Provider object.

        Utilize Keystone URI:
        ``GET /OS-FEDERATION/service_providers/{id}``

        :param service_provider: an object with service_provider_id
                                 stored inside.

        """
        return super(ServiceProviderManager, self).get(service_provider_id=base.getid(service_provider))

    def list(self, **kwargs):
        """List all Service Providers.

        Utilize Keystone URI:
        ``GET /OS-FEDERATION/service_providers``

        """
        return super(ServiceProviderManager, self).list(**kwargs)

    def update(self, service_provider, **kwargs):
        """Update the existing Service Provider object on the server.

        Only properties provided to the function are being updated.

        Utilize Keystone URI:
        ``PATCH /OS-FEDERATION/service_providers/{id}``

        :param service_provider: an object with service_provider_id
                                 stored inside.

        """
        return super(ServiceProviderManager, self).update(service_provider_id=base.getid(service_provider), **kwargs)

    def delete(self, service_provider):
        """Delete Service Provider object.

        Utilize Keystone URI:
        ``DELETE /OS-FEDERATION/service_providers/{id}``

        :param service_provider: an object with service_provider_id
                                 stored inside.

        """
        return super(ServiceProviderManager, self).delete(service_provider_id=base.getid(service_provider))