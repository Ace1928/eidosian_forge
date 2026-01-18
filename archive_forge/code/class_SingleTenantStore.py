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
class SingleTenantStore(BaseStore):
    EXAMPLE_URL = 'swift://<USER>:<KEY>@<AUTH_ADDRESS>/<CONTAINER>/<FILE>'

    def __init__(self, conf, backend=None):
        super(SingleTenantStore, self).__init__(conf, backend=backend)
        self.backend_group = backend
        self.ref_params = sutils.SwiftParams(self.conf, backend=backend).params

    def configure(self, re_raise_bsc=False):
        self.auth_version = self._option_get('swift_store_auth_version')
        self.user_domain_id = None
        self.user_domain_name = None
        self.project_domain_id = None
        self.project_domain_name = None
        super(SingleTenantStore, self).configure(re_raise_bsc=re_raise_bsc)

    def configure_add(self):
        if self.backend_group:
            default_ref = getattr(self.conf, self.backend_group).default_swift_reference
            self.container = getattr(self.conf, self.backend_group).swift_store_container
        else:
            default_ref = self.conf.glance_store.default_swift_reference
            self.container = self.conf.glance_store.swift_store_container
        default_swift_reference = self.ref_params.get(default_ref)
        if default_swift_reference:
            self.auth_address = default_swift_reference.get('auth_address')
        if not default_swift_reference or not self.auth_address:
            reason = _('A value for swift_store_auth_address is required.')
            LOG.error(reason)
            raise exceptions.BadStoreConfiguration(message=reason)
        if self.auth_address.startswith('http://'):
            self.scheme = 'swift+http'
        else:
            self.scheme = 'swift+https'
        self.auth_version = default_swift_reference.get('auth_version')
        self.user = default_swift_reference.get('user')
        self.key = default_swift_reference.get('key')
        self.user_domain_id = default_swift_reference.get('user_domain_id')
        self.user_domain_name = default_swift_reference.get('user_domain_name')
        self.project_domain_id = default_swift_reference.get('project_domain_id')
        self.project_domain_name = default_swift_reference.get('project_domain_name')
        if not (self.user or self.key):
            reason = _('A value for swift_store_ref_params is required.')
            LOG.error(reason)
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        if self.backend_group:
            self._set_url_prefix()

    def _get_credstring(self):
        if self.user and self.key:
            return '%s:%s' % (urllib.parse.quote(self.user), urllib.parse.quote(self.key))
        return ''

    def _set_url_prefix(self, context=None):
        auth_or_store_url = self.auth_address
        if auth_or_store_url.startswith('http://'):
            auth_or_store_url = auth_or_store_url[len('http://'):]
        elif auth_or_store_url.startswith('https://'):
            auth_or_store_url = auth_or_store_url[len('https://'):]
        credstring = self._get_credstring()
        auth_or_store_url = auth_or_store_url.strip('/')
        container = self.container.strip('/')
        if sutils.is_multiple_swift_store_accounts_enabled(self.conf, backend=self.backend_group):
            include_creds = False
        else:
            include_creds = True
        if not include_creds:
            store = getattr(self.conf, self.backend_group).default_swift_reference
            self._url_prefix = '%s://%s/%s/' % ('swift+config', store, container)
            return
        if self.scheme == 'swift+config':
            if self.ssl_enabled:
                self.scheme = 'swift+https'
            else:
                self.scheme = 'swift+http'
        if credstring != '':
            credstring = '%s@' % credstring
        self._url_prefix = '%s://%s%s/%s/' % (self.scheme, credstring, auth_or_store_url, container)

    def create_location(self, image_id, context=None):
        container_name = self.get_container_name(image_id, self.container)
        specs = {'scheme': self.scheme, 'container': container_name, 'obj': str(image_id), 'auth_or_store_url': self.auth_address, 'user': self.user, 'key': self.key}
        return StoreLocation(specs, self.conf, backend_group=self.backend_group)

    def get_container_name(self, image_id, default_image_container):
        """
        Returns appropriate container name depending upon value of
        ``swift_store_multiple_containers_seed``. In single-container mode,
        which is a seed value of 0, simply returns default_image_container.
        In multiple-container mode, returns default_image_container as the
        prefix plus a suffix determined by the multiple container seed

        examples:
            single-container mode:  'glance'
            multiple-container mode: 'glance_3a1' for image uuid 3A1xxxxxxx...

        :param image_id: UUID of image
        :param default_image_container: container name from
               ``swift_store_container``
        """
        if self.backend_group:
            seed_num_chars = getattr(self.conf, self.backend_group).swift_store_multiple_containers_seed
        else:
            seed_num_chars = self.conf.glance_store.swift_store_multiple_containers_seed
        if seed_num_chars is None or seed_num_chars < 0 or seed_num_chars > 32:
            reason = _('An integer value between 0 and 32 is required for swift_store_multiple_containers_seed.')
            LOG.error(reason)
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        elif seed_num_chars > 0:
            image_id = str(image_id).lower()
            num_dashes = image_id[:seed_num_chars].count('-')
            num_chars = seed_num_chars + num_dashes
            name_suffix = image_id[:num_chars]
            new_container_name = default_image_container + '_' + name_suffix
            return new_container_name
        else:
            return default_image_container

    def get_connection(self, location, context=None):
        if not location.user:
            reason = _('Location is missing user:password information.')
            LOG.info(reason)
            raise exceptions.BadStoreUri(message=reason)
        auth_url = location.swift_url
        if not auth_url.endswith('/'):
            auth_url += '/'
        if self.auth_version in ('2', '3'):
            try:
                tenant_name, user = location.user.split(':')
            except ValueError:
                reason = _("Badly formed tenant:user '%(user)s' in Swift URI") % {'user': location.user}
                LOG.info(reason)
                raise exceptions.BadStoreUri(message=reason)
        else:
            tenant_name = None
            user = location.user
        os_options = {}
        if self.region:
            os_options['region_name'] = self.region
        os_options['endpoint_type'] = self.endpoint_type
        os_options['service_type'] = self.service_type
        if self.user_domain_id:
            os_options['user_domain_id'] = self.user_domain_id
        if self.user_domain_name:
            os_options['user_domain_name'] = self.user_domain_name
        if self.project_domain_id:
            os_options['project_domain_id'] = self.project_domain_id
        if self.project_domain_name:
            os_options['project_domain_name'] = self.project_domain_name
        return swiftclient.Connection(auth_url, user, location.key, preauthurl=self.conf_endpoint, insecure=self.insecure, tenant_name=tenant_name, auth_version=self.auth_version, os_options=os_options, ssl_compression=self.ssl_compression, cacert=self.cacert)

    def init_client(self, location, context=None):
        """Initialize keystone client with swift service user credentials"""
        if not location.user:
            reason = _('Location is missing user:password information.')
            LOG.info(reason)
            raise exceptions.BadStoreUri(message=reason)
        auth_url = location.swift_url
        if not auth_url.endswith('/'):
            auth_url += '/'
        try:
            tenant_name, user = location.user.split(':')
        except ValueError:
            reason = _("Badly formed tenant:user '%(user)s' in Swift URI") % {'user': location.user}
            LOG.info(reason)
            raise exceptions.BadStoreUri(message=reason)
        password = ks_identity.V3Password(auth_url=auth_url, username=user, password=location.key, project_name=tenant_name, user_domain_id=self.user_domain_id, user_domain_name=self.user_domain_name, project_domain_id=self.project_domain_id, project_domain_name=self.project_domain_name)
        sess = ks_session.Session(auth=password, verify=self.ks_verify)
        return ks_client.Client(session=sess)

    def get_manager(self, store_location, context=None, allow_reauth=False):
        return connection_manager.SingleTenantConnectionManager(self, store_location, context, allow_reauth)