import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
class MultiTenantStore(BaseStore):
    EXAMPLE_URL = 'swift://<SWIFT_URL>/<CONTAINER>/<FILE>'

    def _get_endpoint(self, context):
        if self.backend_group:
            self.container = getattr(self.conf, self.backend_group).swift_store_container
        else:
            self.container = self.conf.glance_store.swift_store_container
        if context is None:
            reason = _('Multi-tenant Swift storage requires a context.')
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        if context.service_catalog is None:
            reason = _('Multi-tenant Swift storage requires a service catalog.')
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        self.storage_url = self.conf_endpoint
        if not self.storage_url:
            catalog = keystone_sc.ServiceCatalogV2(context.service_catalog)
            self.storage_url = catalog.url_for(service_type=self.service_type, region_name=self.region, interface=self.endpoint_type)
        if self.storage_url.startswith('http://'):
            self.scheme = 'swift+http'
        else:
            self.scheme = 'swift+https'
        return self.storage_url

    def delete(self, location, connection=None, context=None):
        if not connection:
            connection = self.get_connection(location.store_location, context=context)
        super(MultiTenantStore, self).delete(location, connection)
        connection.delete_container(location.store_location.container)

    def set_acls(self, location, public=False, read_tenants=None, write_tenants=None, connection=None, context=None):
        location = location.store_location
        if not connection:
            connection = self.get_connection(location, context=context)
        if read_tenants is None:
            read_tenants = []
        if write_tenants is None:
            write_tenants = []
        headers = {}
        if public:
            headers['X-Container-Read'] = '*:*'
        elif read_tenants:
            headers['X-Container-Read'] = ','.join(('%s:*' % i for i in read_tenants))
        else:
            headers['X-Container-Read'] = ''
        write_tenants.extend(self.admin_tenants)
        if write_tenants:
            headers['X-Container-Write'] = ','.join(('%s:*' % i for i in write_tenants))
        else:
            headers['X-Container-Write'] = ''
        try:
            connection.post_container(location.container, headers=headers)
        except swiftclient.ClientException as e:
            if e.http_status == http.client.NOT_FOUND:
                msg = _('Swift could not find image at URI.')
                raise exceptions.NotFound(message=msg)
            else:
                raise

    def create_location(self, image_id, context=None):
        ep = self._get_endpoint(context)
        specs = {'scheme': self.scheme, 'container': self.container + '_' + str(image_id), 'obj': str(image_id), 'auth_or_store_url': ep}
        return StoreLocation(specs, self.conf, backend_group=self.backend_group)

    def _set_url_prefix(self, context=None):
        ep = self._get_endpoint(context)
        self._url_prefix = '%s://%s:%s_' % (self.scheme, ep, self.container)

    def get_connection(self, location, context=None):
        return swiftclient.Connection(preauthurl=location.swift_url, preauthtoken=context.auth_token, insecure=self.insecure, ssl_compression=self.ssl_compression, cacert=self.cacert)

    def init_client(self, location, context=None):
        ref_params = sutils.SwiftParams(self.conf, backend=self.backend_group).params
        if self.backend_group:
            default_ref = getattr(self.conf, self.backend_group).default_swift_reference
        else:
            default_ref = self.conf.glance_store.default_swift_reference
        default_swift_reference = ref_params.get(default_ref)
        if not default_swift_reference:
            reason = (_('default_swift_reference %s is required.'), default_ref)
            LOG.error(reason)
            raise exceptions.BadStoreConfiguration(message=reason)
        auth_address = default_swift_reference.get('auth_address')
        user = default_swift_reference.get('user')
        key = default_swift_reference.get('key')
        user_domain_id = default_swift_reference.get('user_domain_id')
        user_domain_name = default_swift_reference.get('user_domain_name')
        project_domain_id = default_swift_reference.get('project_domain_id')
        project_domain_name = default_swift_reference.get('project_domain_name')
        if self.backend_group:
            self._set_url_prefix(context=context)
        trustor_auth = ks_identity.V3Token(auth_url=auth_address, token=context.auth_token, project_id=context.project_id)
        trustor_sess = ks_session.Session(auth=trustor_auth, verify=self.ks_verify)
        trustor_client = ks_client.Client(session=trustor_sess)
        auth_ref = trustor_client.session.auth.get_auth_ref(trustor_sess)
        roles = [t['name'] for t in auth_ref['roles']]
        tenant_name, user = user.split(':')
        password = ks_identity.V3Password(auth_url=auth_address, username=user, password=key, project_name=tenant_name, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
        trustee_sess = ks_session.Session(auth=password, verify=self.ks_verify)
        trustee_client = ks_client.Client(session=trustee_sess)
        trustee_user_id = trustee_client.session.get_user_id()
        trust_id = trustor_client.trusts.create(trustee_user=trustee_user_id, trustor_user=context.user_id, project=context.project_id, impersonation=True, role_names=roles).id
        client_password = ks_identity.V3Password(auth_url=auth_address, username=user, password=key, trust_id=trust_id, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
        client_sess = ks_session.Session(auth=client_password, verify=self.ks_verify)
        return ks_client.Client(session=client_sess)

    def get_manager(self, store_location, context=None, allow_reauth=False):
        if self.backend_group:
            use_trusts = getattr(self.conf, self.backend_group).swift_store_use_trusts
        else:
            use_trusts = self.conf.glance_store.swift_store_use_trusts
        if not use_trusts:
            allow_reauth = False
        return connection_manager.MultiTenantConnectionManager(self, store_location, context, allow_reauth)