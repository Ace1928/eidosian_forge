import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
class LegacyLoginHandler(LoginFormHandler):
    """Legacy LoginHandler, implementing most custom auth configuration.

    Deprecated in jupyter-server 2.0.
    Login configuration has moved to IdentityProvider.
    """

    @property
    def hashed_password(self):
        return self.password_from_settings(self.settings)

    def passwd_check(self, a, b):
        """Check a passwd."""
        return passwd_check(a, b)

    @allow_unauthenticated
    def post(self):
        """Post a login form."""
        typed_password = self.get_argument('password', default='')
        new_password = self.get_argument('new_password', default='')
        if self.get_login_available(self.settings):
            if self.passwd_check(self.hashed_password, typed_password) and (not new_password):
                self.set_login_cookie(self, uuid.uuid4().hex)
            elif self.token and self.token == typed_password:
                self.set_login_cookie(self, uuid.uuid4().hex)
                if new_password and getattr(self.identity_provider, 'allow_password_change', False):
                    config_dir = self.settings.get('config_dir', '')
                    config_file = os.path.join(config_dir, 'jupyter_server_config.json')
                    if hasattr(self.identity_provider, 'hashed_password'):
                        self.identity_provider.hashed_password = self.settings['password'] = set_password(new_password, config_file=config_file)
                    self.log.info('Wrote hashed password to %s' % config_file)
            else:
                self.set_status(401)
                self._render(message={'error': 'Invalid credentials'})
                return
        next_url = self.get_argument('next', default=self.base_url)
        self._redirect_safe(next_url)

    @classmethod
    def set_login_cookie(cls, handler, user_id=None):
        """Call this on handlers to set the login cookie for success"""
        cookie_options = handler.settings.get('cookie_options', {})
        cookie_options.setdefault('httponly', True)
        if handler.settings.get('secure_cookie', handler.request.protocol == 'https'):
            cookie_options.setdefault('secure', True)
        cookie_options.setdefault('path', handler.base_url)
        handler.set_secure_cookie(handler.cookie_name, user_id, **cookie_options)
        return user_id
    auth_header_pat = re.compile('token\\s+(.+)', re.IGNORECASE)

    @classmethod
    def get_token(cls, handler):
        """Get the user token from a request

        Default:

        - in URL parameters: ?token=<token>
        - in header: Authorization: token <token>
        """
        user_token = handler.get_argument('token', '')
        if not user_token:
            m = cls.auth_header_pat.match(handler.request.headers.get('Authorization', ''))
            if m:
                user_token = m.group(1)
        return user_token

    @classmethod
    def should_check_origin(cls, handler):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        return not cls.is_token_authenticated(handler)

    @classmethod
    def is_token_authenticated(cls, handler):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        if getattr(handler, '_user_id', None) is None:
            handler.current_user
        return getattr(handler, '_token_authenticated', False)

    @classmethod
    def get_user(cls, handler):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        if getattr(handler, '_user_id', None):
            return handler._user_id
        token_user_id = cls.get_user_token(handler)
        cookie_user_id = cls.get_user_cookie(handler)
        user_id = token_user_id or cookie_user_id
        if token_user_id:
            if user_id != cookie_user_id:
                cls.set_login_cookie(handler, user_id)
            handler._token_authenticated = True
        if user_id is None:
            if handler.get_cookie(handler.cookie_name) is not None:
                handler.log.warning('Clearing invalid/expired login cookie %s', handler.cookie_name)
                handler.clear_login_cookie()
            if not handler.login_available:
                user_id = 'anonymous'
        handler._user_id = user_id
        return user_id

    @classmethod
    def get_user_cookie(cls, handler):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        get_secure_cookie_kwargs = handler.settings.get('get_secure_cookie_kwargs', {})
        user_id = handler.get_secure_cookie(handler.cookie_name, **get_secure_cookie_kwargs)
        if user_id:
            user_id = user_id.decode()
        return user_id

    @classmethod
    def get_user_token(cls, handler):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        token = handler.token
        if not token:
            return None
        user_token = cls.get_token(handler)
        authenticated = False
        if user_token == token:
            handler.log.debug('Accepting token-authenticated connection from %s', handler.request.remote_ip)
            authenticated = True
        if authenticated:
            user_id = cls.get_user_cookie(handler)
            if user_id is None:
                user_id = uuid.uuid4().hex
                handler.log.info(f'Generating new user_id for token-authenticated request: {user_id}')
            return user_id
        else:
            return None

    @classmethod
    def validate_security(cls, app, ssl_options=None):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        if not app.ip:
            warning = 'WARNING: The Jupyter server is listening on all IP addresses'
            if ssl_options is None:
                app.log.warning(f'{warning} and not using encryption. This is not recommended.')
            if not app.password and (not app.token):
                app.log.warning(f'{warning} and not using authentication. This is highly insecure and not recommended.')
        elif not app.password and (not app.token):
            app.log.warning('All authentication is disabled.  Anyone who can connect to this server will be able to run code.')

    @classmethod
    def password_from_settings(cls, settings):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        return settings.get('password', '')

    @classmethod
    def get_login_available(cls, settings):
        """DEPRECATED in 2.0, use IdentityProvider API"""
        return bool(cls.password_from_settings(settings) or settings.get('token'))