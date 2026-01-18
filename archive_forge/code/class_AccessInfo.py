import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
class AccessInfo(object):
    """Encapsulates a raw authentication token from keystone.

    Provides helper methods for extracting useful values from that token.

    """
    _service_catalog_class = None

    def __init__(self, body, auth_token=None):
        self._data = body
        self._auth_token = auth_token
        self._service_catalog = None
        self._service_providers = None

    @property
    def service_catalog(self):
        if not self._service_catalog:
            self._service_catalog = self._service_catalog_class.from_token(self._data)
        return self._service_catalog

    def will_expire_soon(self, stale_duration=STALE_TOKEN_DURATION):
        """Determine if expiration is about to occur.

        :returns: true if expiration is within the given duration
        :rtype: boolean

        """
        norm_expires = utils.normalize_time(self.expires)
        soon = utils.from_utcnow(seconds=stale_duration)
        return norm_expires < soon

    def has_service_catalog(self):
        """Return true if the auth token has a service catalog.

        :returns: boolean
        """
        raise NotImplementedError()

    @property
    def auth_token(self):
        """Return the token_id associated with the auth request.

        To be used in headers for authenticating OpenStack API requests.

        :returns: str
        """
        return self._auth_token

    @property
    def expires(self):
        """Return the token expiration (as datetime object).

        :returns: datetime
        """
        raise NotImplementedError()

    @property
    def issued(self):
        """Return the token issue time (as datetime object).

        :returns: datetime
        """
        raise NotImplementedError()

    @property
    def username(self):
        """Return the username associated with the auth request.

        Follows the pattern defined in the V2 API of first looking for 'name',
        returning that if available, and falling back to 'username' if name
        is unavailable.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def user_id(self):
        """Return the user id associated with the auth request.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def user_domain_id(self):
        """Return the user's domain id associated with the auth request.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def user_domain_name(self):
        """Return the user's domain name associated with the auth request.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def role_ids(self):
        """Return a list of user's role ids associated with the auth request.

        :returns: a list of strings of role ids
        """
        raise NotImplementedError()

    @property
    def role_names(self):
        """Return a list of user's role names associated with the auth request.

        :returns: a list of strings of role names
        """
        raise NotImplementedError()

    @property
    def domain_name(self):
        """Return the domain name associated with the auth request.

        :returns: str or None (if no domain associated with the token)
        """
        raise NotImplementedError()

    @property
    def domain_id(self):
        """Return the domain id associated with the auth request.

        :returns: str or None (if no domain associated with the token)
        """
        raise NotImplementedError()

    @property
    def project_name(self):
        """Return the project name associated with the auth request.

        :returns: str or None (if no project associated with the token)
        """
        raise NotImplementedError()

    @property
    def tenant_name(self):
        """Synonym for project_name."""
        return self.project_name

    @property
    def scoped(self):
        """Return true if the auth token was scoped.

        Returns true if scoped to a tenant(project) or domain,
        and contains a populated service catalog.

        This is deprecated, use project_scoped instead.

        :returns: bool
        """
        return self.project_scoped or self.domain_scoped or self.system_scoped

    @property
    def project_scoped(self):
        """Return true if the auth token was scoped to a tenant (project).

        :returns: bool
        """
        return bool(self.project_id)

    @property
    def domain_scoped(self):
        """Return true if the auth token was scoped to a domain.

        :returns: bool
        """
        raise NotImplementedError()

    @property
    def system_scoped(self):
        """Return true if the auth token was scoped to the system.

        :returns: bool
        """
        raise NotImplementedError()

    @property
    def trust_id(self):
        """Return the trust id associated with the auth request.

        :returns: str or None (if no trust associated with the token)
        """
        raise NotImplementedError()

    @property
    def trust_scoped(self):
        """Return true if the auth token was scoped from a delegated trust.

        The trust delegation is via the OS-TRUST v3 extension.

        :returns: bool
        """
        raise NotImplementedError()

    @property
    def trustee_user_id(self):
        """Return the trustee user id associated with a trust.

        :returns: str or None (if no trust associated with the token)
        """
        raise NotImplementedError()

    @property
    def trustor_user_id(self):
        """Return the trustor user id associated with a trust.

        :returns: str or None (if no trust associated with the token)
        """
        raise NotImplementedError()

    @property
    def project_id(self):
        """Return the project ID associated with the auth request.

        This returns None if the auth token wasn't scoped to a project.

        :returns: str or None (if no project associated with the token)
        """
        raise NotImplementedError()

    @property
    def tenant_id(self):
        """Synonym for project_id."""
        return self.project_id

    @property
    def project_domain_id(self):
        """Return the project's domain id associated with the auth request.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def project_domain_name(self):
        """Return the project's domain name associated with the auth request.

        :returns: str
        """
        raise NotImplementedError()

    @property
    def oauth_access_token_id(self):
        """Return the access token ID if OAuth authentication used.

        :returns: str or None.
        """
        raise NotImplementedError()

    @property
    def oauth_consumer_id(self):
        """Return the consumer ID if OAuth authentication used.

        :returns: str or None.
        """
        raise NotImplementedError()

    @property
    def is_federated(self):
        """Return true if federation was used to get the token.

        :returns: boolean
        """
        raise NotImplementedError()

    @property
    def is_admin_project(self):
        """Return true if the current project scope is the admin project.

        For backwards compatibility purposes if there is nothing specified in
        the token we always assume we are in the admin project, so this will
        default to True.

        :returns boolean
        """
        raise NotImplementedError()

    @property
    def audit_id(self):
        """Return the audit ID if present.

        :returns: str or None.
        """
        raise NotImplementedError()

    @property
    def audit_chain_id(self):
        """Return the audit chain ID if present.

        In the event that a token was rescoped then this ID will be the
        :py:attr:`audit_id` of the initial token. Returns None if no value
        present.

        :returns: str or None.
        """
        raise NotImplementedError()

    @property
    def initial_audit_id(self):
        """The audit ID of the initially requested token.

        This is the :py:attr:`audit_chain_id` if present or the
        :py:attr:`audit_id`.
        """
        return self.audit_chain_id or self.audit_id

    @property
    def service_providers(self):
        """Return an object representing the list of trusted service providers.

        Used for Keystone2Keystone federating-out.

        :returns: :py:class:`keystoneauth1.service_providers.ServiceProviders`
                  or None
        """
        raise NotImplementedError()

    @property
    def bind(self):
        """Information about external mechanisms the token is bound to.

        If a token is bound to an external authentication mechanism it can only
        be used in conjunction with that mechanism. For example if bound to a
        kerberos principal it may only be accepted if there is also kerberos
        authentication performed on the request.

        :returns: A dictionary or None. The key will be the bind type the value
                  is a dictionary that is specific to the format of the bind
                  type. Returns None if there is no bind information in the
                  token.
        """
        raise NotImplementedError()

    @property
    def project_is_domain(self):
        """Return if a project act as a domain.

        :returns: bool
        """
        raise NotImplementedError()