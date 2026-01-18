from barbicanclient import exceptions
from barbicanclient.v1 import client as barbican_client
from barbicanclient.v1 import containers
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class BarbicanClientPlugin(client_plugin.ClientPlugin):
    service_types = [KEY_MANAGER] = ['key-manager']

    def _create(self):
        interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        client = barbican_client.Client(session=self.context.keystone_session, service_type=self.KEY_MANAGER, interface=interface, connect_retries=cfg.CONF.client_retry_limit, region_name=self._get_region_name())
        return client

    def is_not_found(self, ex):
        return isinstance(ex, exceptions.HTTPClientError) and ex.status_code == 404

    def create_generic_container(self, **props):
        return containers.Container(self.client().containers._api, **props)

    def create_certificate(self, **props):
        return containers.CertificateContainer(self.client().containers._api, **props)

    def create_rsa(self, **props):
        return containers.RSAContainer(self.client().containers._api, **props)

    def get_secret_by_ref(self, secret_ref):
        try:
            secret = self.client().secrets.get(secret_ref)
            secret.name
            return secret
        except Exception as ex:
            if self.is_not_found(ex):
                raise exception.EntityNotFound(entity='Secret', name=secret_ref)
            LOG.info('Failed to get Barbican secret from reference %s' % secret_ref)
            raise

    def get_secret_payload_by_ref(self, secret_ref):
        return self.get_secret_by_ref(secret_ref).payload

    def get_container_by_ref(self, container_ref):
        try:
            return self.client().containers.get(container_ref)
        except Exception as ex:
            if self.is_not_found(ex):
                raise exception.EntityNotFound(entity='Container', name=container_ref)
            raise