from the request environment and it's identified by the ``swift.cache`` key.
import copy
import re
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.auth_token import _identity
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.auth_token import _request
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.i18n import _
class BaseAuthProtocol(object):
    """A base class for AuthProtocol token checking implementations.

    :param Callable app: The next application to call after middleware.
    :param logging.Logger log: The logging object to use for output. By default
                               it will use a logger in the
                               keystonemiddleware.auth_token namespace.
    :param str enforce_token_bind: The style of token binding enforcement to
                                   perform.
    """

    def __init__(self, app, log=_LOG, enforce_token_bind=_BIND_MODE.PERMISSIVE, service_token_roles=None, service_token_roles_required=False, service_type=None):
        self.log = log
        self._app = app
        self._enforce_token_bind = enforce_token_bind
        self._service_token_roles = set(service_token_roles or [])
        self._service_token_roles_required = service_token_roles_required
        self._service_token_warning_emitted = False
        self._service_type = service_type

    @webob.dec.wsgify(RequestClass=_request._AuthTokenRequest)
    def __call__(self, req):
        """Handle incoming request."""
        response = self.process_request(req)
        if response:
            return response
        response = req.get_response(self._app)
        return self.process_response(response)

    def process_request(self, request):
        """Process request.

        If this method returns a value then that value will be used as the
        response. The next application down the stack will not be executed and
        process_response will not be called.

        Otherwise, the next application down the stack will be executed and
        process_response will be called with the generated response.

        By default this method does not return a value.

        :param request: Incoming request
        :type request: _request.AuthTokenRequest

        """
        user_auth_ref = None
        serv_auth_ref = None
        allow_expired = False
        if request.service_token:
            self.log.debug('Authenticating service token')
            try:
                _, serv_auth_ref = self._do_fetch_token(request.service_token)
                self._validate_token(serv_auth_ref)
                self._confirm_token_bind(serv_auth_ref, request)
            except ksm_exceptions.InvalidToken:
                self.log.info('Invalid service token')
                request.service_token_valid = False
            else:
                role_names = set(serv_auth_ref.role_names)
                check = self._service_token_roles.intersection(role_names)
                role_check_passed = bool(check)
                if self._service_token_roles_required:
                    request.service_token_valid = role_check_passed
                else:
                    if not self._service_token_warning_emitted:
                        self.log.warning('A valid token was submitted as a service token, but it was not a valid service token. This is incorrect but backwards compatible behaviour. This will be removed in future releases.')
                        self._service_token_warning_emitted = True
                    request.service_token_valid = True
                allow_expired = role_check_passed
        if request.user_token:
            self.log.debug('Authenticating user token')
            try:
                data, user_auth_ref = self._do_fetch_token(request.user_token, allow_expired=allow_expired)
                self._validate_token(user_auth_ref, allow_expired=allow_expired)
                if user_auth_ref.version != 'v2.0':
                    self.validate_allowed_request(request, data['token'])
                if not request.service_token:
                    self._confirm_token_bind(user_auth_ref, request)
            except ksm_exceptions.InvalidToken:
                self.log.info('Invalid user token')
                request.user_token_valid = False
            else:
                request.user_token_valid = True
                request.token_info = data
        request.token_auth = _user_plugin.UserAuthPlugin(user_auth_ref, serv_auth_ref)

    def _validate_token(self, auth_ref, allow_expired=False):
        """Perform the validation steps on the token.

        :param auth_ref: The token data
        :type auth_ref: keystoneauth1.access.AccessInfo

        :raises exc.InvalidToken: if token is rejected
        """
        if not allow_expired and auth_ref.will_expire_soon(stale_duration=0):
            raise ksm_exceptions.InvalidToken(_('Token authorization failed'))

    def _do_fetch_token(self, token, **kwargs):
        """Helper method to fetch a token and convert it into an AccessInfo."""
        token = token.strip()
        data = self.fetch_token(token, **kwargs)
        try:
            return (data, access.create(body=data, auth_token=token))
        except Exception:
            self.log.warning('Invalid token contents.', exc_info=True)
            raise ksm_exceptions.InvalidToken(_('Token authorization failed'))

    def fetch_token(self, token, **kwargs):
        """Fetch the token data based on the value in the header.

        Retrieve the data associated with the token value that was in the
        header. This can be from PKI, contacting the identity server or
        whatever is required.

        :param str token: The token present in the request header.
        :param dict kwargs: Additional keyword arguments may be passed through
                            here to support new features. If an implementation
                            is not aware of how to use these arguments it
                            should ignore them.

        :raises exc.InvalidToken: if token is invalid.

        :returns: The token data
        :rtype: dict
        """
        raise NotImplementedError()

    def process_response(self, response):
        """Do whatever you'd like to the response.

        By default the response is returned unmodified.

        :param response: Response object
        :type response: ._request._AuthTokenResponse
        """
        return response

    def _invalid_user_token(self, msg=False):
        if msg is False:
            msg = _('Token authorization failed')
        raise ksm_exceptions.InvalidToken(msg)

    def _confirm_token_bind(self, auth_ref, req):
        if self._enforce_token_bind == _BIND_MODE.DISABLED:
            return
        permissive = self._enforce_token_bind in (_BIND_MODE.PERMISSIVE, _BIND_MODE.STRICT)
        if not auth_ref.bind:
            if permissive:
                return
            else:
                self.log.info('No bind information present in token.')
                self._invalid_user_token()
        if permissive or self._enforce_token_bind == _BIND_MODE.REQUIRED:
            name = None
        else:
            name = self._enforce_token_bind
        if name and name not in auth_ref.bind:
            self.log.info('Named bind mode %s not in bind information', name)
            self._invalid_user_token()
        for bind_type, identifier in auth_ref.bind.items():
            if bind_type == _BIND_MODE.KERBEROS:
                if req.auth_type != 'negotiate':
                    self.log.info('Kerberos credentials required and not present.')
                    self._invalid_user_token()
                if req.remote_user != identifier:
                    self.log.info('Kerberos credentials do not match those in bind.')
                    self._invalid_user_token()
                self.log.debug('Kerberos bind authentication successful.')
            elif self._enforce_token_bind == _BIND_MODE.PERMISSIVE:
                self.log.debug('Ignoring Unknown bind for permissive mode: %(bind_type)s: %(identifier)s.', {'bind_type': bind_type, 'identifier': identifier})
            else:
                self.log.info('Couldn`t verify unknown bind: %(bind_type)s: %(identifier)s.', {'bind_type': bind_type, 'identifier': identifier})
                self._invalid_user_token()

    def validate_allowed_request(self, request, token):
        self.log.debug('Validating token access rules against request')
        app_cred = token.get('application_credential')
        if not app_cred:
            return
        access_rules = app_cred.get('access_rules')
        if access_rules is None:
            return
        if hasattr(self, '_conf'):
            my_service_type = self._conf.get('service_type')
        else:
            my_service_type = self._service_type
        if not my_service_type:
            self.log.warning('Cannot validate request with restricted access rules. Set service_type in [keystone_authtoken] to allow access rule validation.')
            raise ksm_exceptions.InvalidToken(_('Token authorization failed'))
        if my_service_type == 'identity' and request.method == 'GET' and request.path.endswith('/v3/auth/tokens'):
            return
        catalog = token['catalog']
        catalog_svcs = [s for s in catalog if s['type'] == my_service_type]
        if len(catalog_svcs) == 0:
            self.log.warning('Cannot validate request with restricted access rules. service_type in [keystone_authtoken] is not a valid service type in the catalog.')
            raise ksm_exceptions.InvalidToken(_('Token authorization failed'))
        if request.service_token:
            return
        for access_rule in access_rules:
            method = access_rule['method']
            path = access_rule['path']
            service = access_rule['service']
            if request.method == method and service == my_service_type and _path_matches(request.path, path):
                return
        raise ksm_exceptions.InvalidToken(_('Token authorization failed'))