import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
class KsClientWrapper(object):
    """Wrap keystone client so we can encapsulate logic used in resources.

    Note this is intended to be initialized from a resource on a per-session
    basis, so the session context is passed in on initialization
    Also note that an instance of this is created in each request context as
    part of a lazy-loaded cloud backend and it can be easily referenced in
    each resource as ``self.keystone()``, so there should not be any need to
    directly instantiate instances of this class inside resources themselves.
    """

    def __init__(self, context, region_name):
        self._context = weakref.ref(context)
        self._client = None
        self._admin_auth = None
        self._domain_admin_auth = None
        self._domain_admin_client = None
        self._region_name = region_name
        self._interface = config.get_client_option('keystone', 'endpoint_type')
        self.session = self.context.keystone_session
        self.v3_endpoint = self.context.keystone_v3_endpoint
        if self.context.trust_id:
            self._client = self._v3_client_init()
        self._stack_domain_id = cfg.CONF.stack_user_domain_id
        self.stack_domain_name = cfg.CONF.stack_user_domain_name
        self.domain_admin_user = cfg.CONF.stack_domain_admin
        self.domain_admin_password = cfg.CONF.stack_domain_admin_password
        LOG.debug('Using stack domain %s', self.stack_domain)

    @property
    def context(self):
        ctxt = self._context()
        assert ctxt is not None, 'Need a reference to the context'
        return ctxt

    @property
    def stack_domain(self):
        """Domain scope data.

        This is only used for checking for scoping data, not using the value.
        """
        return self._stack_domain_id or self.stack_domain_name

    @property
    def client(self):
        if not self._client:
            self._client = self._v3_client_init()
        return self._client

    @property
    def auth_region_name(self):
        importutils.import_module('keystonemiddleware.auth_token')
        auth_region = cfg.CONF.keystone_authtoken.region_name
        if not auth_region:
            auth_region = self._region_name
        return auth_region

    @property
    def domain_admin_auth(self):
        if not self._domain_admin_auth:
            auth = ks_auth.Password(username=self.domain_admin_user, password=self.domain_admin_password, auth_url=self.v3_endpoint, domain_id=self._stack_domain_id, domain_name=self.stack_domain_name, user_domain_id=self._stack_domain_id, user_domain_name=self.stack_domain_name)
            try:
                auth.get_token(self.session)
            except ks_exception.Unauthorized:
                LOG.error('Domain admin client authentication failed')
                raise exception.AuthorizationFailure()
            self._domain_admin_auth = auth
        return self._domain_admin_auth

    @property
    def domain_admin_client(self):
        if not self._domain_admin_client:
            self._domain_admin_client = kc_v3.Client(session=self.session, auth=self.domain_admin_auth, connect_retries=cfg.CONF.client_retry_limit, interface=self._interface, region_name=self.auth_region_name)
        return self._domain_admin_client

    def _v3_client_init(self):
        client = kc_v3.Client(session=self.session, connect_retries=cfg.CONF.client_retry_limit, interface=self._interface, region_name=self.auth_region_name)
        if hasattr(self.context.auth_plugin, 'get_access'):
            try:
                auth_ref = self.context.auth_plugin.get_access(self.session)
            except ks_exception.Unauthorized:
                LOG.error('Keystone client authentication failed')
                raise exception.AuthorizationFailure()
            if self.context.trust_id:
                if not auth_ref.trust_scoped:
                    LOG.error('trust token re-scoping failed!')
                    raise exception.AuthorizationFailure()
                if self.context.trustor_user_id != auth_ref.user_id:
                    LOG.error('Trust impersonation failed')
                    raise exception.AuthorizationFailure()
        return client

    def _create_trust_context(self, trustor_user_id, trustor_proj_id):
        try:
            trustee_user_id = self.context.trusts_auth_plugin.get_user_id(self.session)
        except ks_exception.Unauthorized:
            LOG.error('Domain admin client authentication failed')
            raise exception.AuthorizationFailure()
        role_kw = {}
        if cfg.CONF.trusts_delegated_roles:
            role_kw['role_names'] = cfg.CONF.trusts_delegated_roles
        else:
            token_info = self.context.auth_token_info
            if token_info and token_info.get('token', {}).get('roles'):
                role_kw['role_ids'] = [r['id'] for r in token_info['token']['roles']]
            else:
                role_kw['role_names'] = self.context.roles
        allow_redelegation = cfg.CONF.reauthentication_auth_method == 'trusts' and cfg.CONF.allow_trusts_redelegation
        try:
            trust = self.client.trusts.create(trustor_user=trustor_user_id, trustee_user=trustee_user_id, project=trustor_proj_id, impersonation=True, allow_redelegation=allow_redelegation, **role_kw)
        except ks_exception.NotFound:
            LOG.debug('Failed to find roles %s for user %s' % (role_kw, trustor_user_id))
            raise exception.MissingCredentialError(required=_('roles %s') % role_kw)
        context_data = self.context.to_dict()
        context_data['overwrite'] = False
        trust_context = context.RequestContext.from_dict(context_data)
        trust_context.trust_id = trust.id
        trust_context.trustor_user_id = trustor_user_id
        return trust_context

    def create_trust_context(self):
        """Create a trust using the trustor identity in the current context.

        The trust is created with the trustee as the heat service user.

        If the current context already contains a trust_id, we do nothing
        and return the current context.

        Returns a context containing the new trust_id.
        """
        if self.context.trust_id:
            return self.context
        trustor_user_id = self.context.auth_plugin.get_user_id(self.session)
        trustor_proj_id = self.context.auth_plugin.get_project_id(self.session)
        return self._create_trust_context(trustor_user_id, trustor_proj_id)

    def delete_trust(self, trust_id):
        """Delete the specified trust."""
        try:
            self.client.trusts.delete(trust_id)
        except (ks_exception.NotFound, ks_exception.Unauthorized):
            pass

    def regenerate_trust_context(self):
        """Regenerate a trust using the trustor identity of current user_id.

        The trust is created with the trustee as the heat service user.

        Returns a context containing the new trust_id.
        """
        old_trust_id = self.context.trust_id
        trustor_user_id = self.context.auth_plugin.get_user_id(self.session)
        trustor_proj_id = self.context.auth_plugin.get_project_id(self.session)
        trust_context = self._create_trust_context(trustor_user_id, trustor_proj_id)
        if old_trust_id:
            self.delete_trust(old_trust_id)
        return trust_context

    def _get_username(self, username):
        if len(username) > 255:
            LOG.warning('Truncating the username %s to the last 255 characters.', username)
        return username[-255:]

    def create_stack_user(self, username, password):
        """Create a user defined as part of a stack.

        The user is defined either via template or created internally by a
        resource.  This user will be added to the heat_stack_user_role as
        defined in the config.

        Returns the keystone ID of the resulting user.
        """
        stack_user_role = self.client.roles.list(name=cfg.CONF.heat_stack_user_role)
        if len(stack_user_role) == 1:
            role_id = stack_user_role[0].id
            user = self.client.users.create(name=self._get_username(username), password=password, default_project=self.context.tenant_id)
            LOG.debug('Adding user %(user)s to role %(role)s', {'user': user.id, 'role': role_id})
            self.client.roles.grant(role=role_id, user=user.id, project=self.context.tenant_id)
        else:
            LOG.error('Failed to add user %(user)s to role %(role)s, check role exists!', {'user': username, 'role': cfg.CONF.heat_stack_user_role})
            raise exception.Error(_("Can't find role %s") % cfg.CONF.heat_stack_user_role)
        return user.id

    def stack_domain_user_token(self, user_id, project_id, password):
        """Get a token for a stack domain user."""
        if not self.stack_domain:
            msg = _('Cannot get stack domain user token, no stack domain id configured, please fix your heat.conf')
            raise exception.Error(msg)
        auth = ks_auth.Password(auth_url=self.v3_endpoint, user_id=user_id, password=password, project_id=project_id)
        return auth.get_token(self.session)

    def create_stack_domain_user(self, username, project_id, password=None):
        """Create a domain user defined as part of a stack.

        The user is defined either via template or created internally by a
        resource.  This user will be added to the heat_stack_user_role as
        defined in the config, and created in the specified project (which is
        expected to be in the stack_domain).

        Returns the keystone ID of the resulting user.
        """
        if not self.stack_domain:
            return self.create_stack_user(username=username, password=password)
        user_options = {'ignore_change_password_upon_first_use': True, 'ignore_password_expiry': True, 'ignore_lockout_failure_attempts': True}
        stack_user_role = self.domain_admin_client.roles.list(name=cfg.CONF.heat_stack_user_role)
        if len(stack_user_role) == 1:
            role_id = stack_user_role[0].id
            user = self.domain_admin_client.users.create(name=self._get_username(username), password=password, default_project=project_id, domain=self.stack_domain_id, options=user_options)
            LOG.debug('Adding user %(user)s to role %(role)s', {'user': user.id, 'role': role_id})
            self.domain_admin_client.roles.grant(role=role_id, user=user.id, project=project_id)
        else:
            LOG.error('Failed to add user %(user)s to role %(role)s, check role exists!', {'user': username, 'role': cfg.CONF.heat_stack_user_role})
            raise exception.Error(_("Can't find role %s") % cfg.CONF.heat_stack_user_role)
        return user.id

    @property
    def stack_domain_id(self):
        if not self._stack_domain_id:
            try:
                access = self.domain_admin_auth.get_access(self.session)
            except ks_exception.Unauthorized:
                LOG.error('Keystone client authentication failed')
                raise exception.AuthorizationFailure()
            self._stack_domain_id = access.domain_id
        return self._stack_domain_id

    def _check_stack_domain_user(self, user_id, project_id, action):
        """Sanity check that domain/project is correct."""
        user = self.domain_admin_client.users.get(user_id)
        if user.domain_id != self.stack_domain_id:
            raise ValueError(_('User %s in invalid domain') % action)
        if user.default_project_id != project_id:
            raise ValueError(_('User %s in invalid project') % action)

    def delete_stack_domain_user(self, user_id, project_id):
        if not self.stack_domain:
            return self.delete_stack_user(user_id)
        try:
            self._check_stack_domain_user(user_id, project_id, 'delete')
            self.domain_admin_client.users.delete(user_id)
        except ks_exception.NotFound:
            pass

    def delete_stack_user(self, user_id):
        try:
            self.client.users.delete(user=user_id)
        except ks_exception.NotFound:
            pass

    def create_stack_domain_project(self, stack_id):
        """Create a project in the heat stack-user domain."""
        if not self.stack_domain:
            return self.context.tenant_id
        project_name = ('%s-%s' % (self.context.tenant_id, stack_id))[:64]
        desc = 'Heat stack user project'
        domain_project = self.domain_admin_client.projects.create(name=project_name, domain=self.stack_domain_id, description=desc)
        return domain_project.id

    def delete_stack_domain_project(self, project_id):
        if not self.stack_domain:
            return
        try:
            project = self.domain_admin_client.projects.get(project=project_id)
        except ks_exception.NotFound:
            return
        except ks_exception.Forbidden:
            LOG.warning('Unable to get details for project %s, not deleting', project_id)
            return
        if project.domain_id != self.stack_domain_id:
            LOG.warning('Not deleting non heat-domain project')
            return
        try:
            project.delete()
        except ks_exception.NotFound:
            pass

    def _find_ec2_keypair(self, access, user_id=None):
        """Lookup an ec2 keypair by access ID."""
        credentials = self.client.credentials.list()
        for cr in credentials:
            ec2_creds = jsonutils.loads(cr.blob)
            if ec2_creds.get('access') == access:
                return AccessKey(id=cr.id, access=ec2_creds['access'], secret=ec2_creds['secret'])

    def delete_ec2_keypair(self, credential_id=None, access=None, user_id=None):
        """Delete credential containing ec2 keypair."""
        if credential_id:
            try:
                self.client.credentials.delete(credential_id)
            except ks_exception.NotFound:
                pass
        elif access:
            cred = self._find_ec2_keypair(access=access, user_id=user_id)
            if cred:
                self.client.credentials.delete(cred.id)
        else:
            raise ValueError('Must specify either credential_id or access')

    def get_ec2_keypair(self, credential_id=None, access=None, user_id=None):
        """Get an ec2 keypair via v3/credentials, by id or access."""
        if credential_id:
            cred = self.client.credentials.get(credential_id)
            ec2_creds = jsonutils.loads(cred.blob)
            return AccessKey(id=cred.id, access=ec2_creds['access'], secret=ec2_creds['secret'])
        elif access:
            return self._find_ec2_keypair(access=access, user_id=user_id)
        else:
            raise ValueError('Must specify either credential_id or access')

    def create_ec2_keypair(self, user_id=None):
        user_id = user_id or self.context.get_access(self.session).user_id
        project_id = self.context.tenant_id
        data_blob = {'access': uuid.uuid4().hex, 'secret': password_gen.generate_openstack_password()}
        ec2_creds = self.client.credentials.create(user=user_id, type='ec2', blob=jsonutils.dumps(data_blob), project=project_id)
        return AccessKey(id=ec2_creds.id, access=data_blob['access'], secret=data_blob['secret'])

    def create_stack_domain_user_keypair(self, user_id, project_id):
        if not self.stack_domain:
            return self.create_ec2_keypair(user_id)
        data_blob = {'access': uuid.uuid4().hex, 'secret': password_gen.generate_openstack_password()}
        creds = self.domain_admin_client.credentials.create(user=user_id, type='ec2', blob=jsonutils.dumps(data_blob), project=project_id)
        return AccessKey(id=creds.id, access=data_blob['access'], secret=data_blob['secret'])

    def delete_stack_domain_user_keypair(self, user_id, project_id, credential_id):
        if not self.stack_domain:
            return self.delete_ec2_keypair(credential_id=credential_id)
        self._check_stack_domain_user(user_id, project_id, 'delete_keypair')
        try:
            self.domain_admin_client.credentials.delete(credential_id)
        except ks_exception.NotFound:
            pass

    def disable_stack_user(self, user_id):
        self.client.users.update(user=user_id, enabled=False)

    def enable_stack_user(self, user_id):
        self.client.users.update(user=user_id, enabled=True)

    def disable_stack_domain_user(self, user_id, project_id):
        if not self.stack_domain:
            return self.disable_stack_user(user_id)
        self._check_stack_domain_user(user_id, project_id, 'disable')
        self.domain_admin_client.users.update(user=user_id, enabled=False)

    def enable_stack_domain_user(self, user_id, project_id):
        if not self.stack_domain:
            return self.enable_stack_user(user_id)
        self._check_stack_domain_user(user_id, project_id, 'enable')
        self.domain_admin_client.users.update(user=user_id, enabled=True)

    def server_keystone_endpoint_url(self, fallback_endpoint):
        ks_endpoint_type = cfg.CONF.server_keystone_endpoint_type
        if ks_endpoint_type == 'public' or ks_endpoint_type == 'internal' or ks_endpoint_type == 'admin':
            if hasattr(self.context, 'auth_plugin') and hasattr(self.context.auth_plugin, 'get_access'):
                try:
                    auth_ref = self.context.auth_plugin.get_access(self.session)
                    if hasattr(auth_ref, 'service_catalog'):
                        unversioned_sc_auth_uri = auth_ref.service_catalog.get_urls(service_type='identity', interface=ks_endpoint_type)
                        if len(unversioned_sc_auth_uri) > 0:
                            sc_auth_uri = unversioned_sc_auth_uri[0] + '/v3'
                            return sc_auth_uri
                except ks_exception.Unauthorized:
                    LOG.error('Keystone client authentication failed')
        return fallback_endpoint