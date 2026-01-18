import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class AuthenticationConfig:
    """The authentication configuration file based on a ini file.

    Implements the authentication.conf file described in
    doc/developers/authentication-ring.txt.
    """

    def __init__(self, _file=None):
        self._config = None
        if _file is None:
            self._input = self._filename = bedding.authentication_config_path()
            self._check_permissions()
        else:
            self._filename = None
            self._input = _file

    def _get_config(self):
        if self._config is not None:
            return self._config
        try:
            self._config = ConfigObj(self._input, encoding='utf-8')
        except configobj.ConfigObjError as e:
            raise ParseConfigError(e.errors, e.config.filename)
        except UnicodeError:
            raise ConfigContentError(self._filename)
        return self._config

    def _check_permissions(self):
        """Check permission of auth file are user read/write able only."""
        import stat
        try:
            st = os.stat(self._filename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                trace.mutter('Unable to stat %r: %r', self._filename, e)
            return
        mode = stat.S_IMODE(st.st_mode)
        if (stat.S_IXOTH | stat.S_IWOTH | stat.S_IROTH | stat.S_IXGRP | stat.S_IWGRP | stat.S_IRGRP) & mode:
            if self._filename not in _authentication_config_permission_errors and (not GlobalConfig().suppress_warning('insecure_permissions')):
                trace.warning("The file '%s' has insecure file permissions. Saved passwords may be accessible by other users.", self._filename)
                _authentication_config_permission_errors.add(self._filename)

    def _save(self):
        """Save the config file, only tests should use it for now."""
        conf_dir = os.path.dirname(self._filename)
        bedding.ensure_config_dir_exists(conf_dir)
        fd = os.open(self._filename, os.O_RDWR | os.O_CREAT, 384)
        try:
            f = os.fdopen(fd, 'wb')
            self._get_config().write(f)
        finally:
            f.close()

    def _set_option(self, section_name, option_name, value):
        """Set an authentication configuration option"""
        conf = self._get_config()
        section = conf.get(section_name)
        if section is None:
            conf[section_name] = {}
            section = conf[section_name]
        section[option_name] = value
        self._save()

    def get_credentials(self, scheme, host, port=None, user=None, path=None, realm=None):
        """Returns the matching credentials from authentication.conf file.

        Args:
          scheme: protocol
          host: the server address
          port: the associated port (optional)
          user: login (optional)
          path: the absolute path on the server (optional)
          realm: the http authentication realm (optional)

        Returns:
          A dict containing the matching credentials or None.
          This includes:
           - name: the section name of the credentials in the
             authentication.conf file,
           - user: can't be different from the provided user if any,
           - scheme: the server protocol,
           - host: the server address,
           - port: the server port (can be None),
           - path: the absolute server path (can be None),
           - realm: the http specific authentication realm (can be None),
           - password: the decoded password, could be None if the credential
             defines only the user
           - verify_certificates: https specific, True if the server
             certificate should be verified, False otherwise.
        """
        credentials = None
        for auth_def_name, auth_def in self._get_config().iteritems():
            if not isinstance(auth_def, configobj.Section):
                raise ValueError('%s defined outside a section' % auth_def_name)
            a_scheme, a_host, a_user, a_path = map(auth_def.get, ['scheme', 'host', 'user', 'path'])
            try:
                a_port = auth_def.as_int('port')
            except KeyError:
                a_port = None
            except ValueError:
                raise ValueError("'port' not numeric in %s" % auth_def_name)
            try:
                a_verify_certificates = auth_def.as_bool('verify_certificates')
            except KeyError:
                a_verify_certificates = True
            except ValueError:
                raise ValueError("'verify_certificates' not boolean in %s" % auth_def_name)
            if a_scheme is not None and scheme != a_scheme:
                continue
            if a_host is not None:
                if not (host == a_host or (a_host.startswith('.') and host.endswith(a_host))):
                    continue
            if a_port is not None and port != a_port:
                continue
            if a_path is not None and path is not None and (not path.startswith(a_path)):
                continue
            if a_user is not None and user is not None and (a_user != user):
                continue
            if a_user is None:
                continue
            credentials = dict(name=auth_def_name, user=a_user, scheme=a_scheme, host=host, port=port, path=path, realm=realm, password=auth_def.get('password', None), verify_certificates=a_verify_certificates)
            self.decode_password(credentials, auth_def.get('password_encoding', None))
            if 'auth' in debug.debug_flags:
                trace.mutter('Using authentication section: %r', auth_def_name)
            break
        if credentials is None:
            credentials = credential_store_registry.get_fallback_credentials(scheme, host, port, user, path, realm)
        return credentials

    def set_credentials(self, name, host, user, scheme=None, password=None, port=None, path=None, verify_certificates=None, realm=None):
        """Set authentication credentials for a host.

        Any existing credentials with matching scheme, host, port and path
        will be deleted, regardless of name.

        Args:
          name: An arbitrary name to describe this set of credentials.
          host: Name of the host that accepts these credentials.
          user: The username portion of these credentials.
          scheme: The URL scheme (e.g. ssh, http) the credentials apply to.
          password: Password portion of these credentials.
          port: The IP port on the host that these credentials apply to.
          path: A filesystem path on the host that these credentials apply to.
          verify_certificates: On https, verify server certificates if True.
          realm: The http authentication realm (optional).
        """
        values = {'host': host, 'user': user}
        if password is not None:
            values['password'] = password
        if scheme is not None:
            values['scheme'] = scheme
        if port is not None:
            values['port'] = '%d' % port
        if path is not None:
            values['path'] = path
        if verify_certificates is not None:
            values['verify_certificates'] = str(verify_certificates)
        if realm is not None:
            values['realm'] = realm
        config = self._get_config()
        for section, existing_values in config.iteritems():
            for key in ('scheme', 'host', 'port', 'path', 'realm'):
                if existing_values.get(key) != values.get(key):
                    break
            else:
                del config[section]
        config.update({name: values})
        self._save()

    def get_user(self, scheme, host, port=None, realm=None, path=None, prompt=None, ask=False, default=None):
        """Get a user from authentication file.

        Args:
          scheme: protocol
          host: the server address
          port: the associated port (optional)
          realm: the realm sent by the server (optional)
          path: the absolute path on the server (optional)
          ask: Ask the user if there is no explicitly configured username
                    (optional)
          default: The username returned if none is defined (optional).

        Returns:
          The found user.
        """
        credentials = self.get_credentials(scheme, host, port, user=None, path=path, realm=realm)
        if credentials is not None:
            user = credentials['user']
        else:
            user = None
        if user is None:
            if ask:
                if prompt is None:
                    prompt = '{}'.format(scheme.upper()) + ' %(host)s username'
                if port is not None:
                    prompt_host = '%s:%d' % (host, port)
                else:
                    prompt_host = host
                user = ui.ui_factory.get_username(prompt, host=prompt_host)
            else:
                user = default
        return user

    def get_password(self, scheme, host, user, port=None, realm=None, path=None, prompt=None):
        """Get a password from authentication file or prompt the user for one.

        Args:
          scheme: protocol
          host: the server address
          port: the associated port (optional)
          user: login
          realm: the realm sent by the server (optional)
          path: the absolute path on the server (optional)

        Returns:
          The found password or the one entered by the user.
        """
        credentials = self.get_credentials(scheme, host, port, user, path, realm)
        if credentials is not None:
            password = credentials['password']
            if password is not None and scheme == 'ssh':
                trace.warning('password ignored in section [%s], use an ssh agent instead' % credentials['name'])
                password = None
        else:
            password = None
        if password is None:
            if prompt is None:
                prompt = '%s' % scheme.upper() + ' %(user)s@%(host)s password'
            if port is not None:
                prompt_host = '%s:%d' % (host, port)
            else:
                prompt_host = host
            password = ui.ui_factory.get_password(prompt, host=prompt_host, user=user)
        return password

    def decode_password(self, credentials, encoding):
        try:
            cs = credential_store_registry.get_credential_store(encoding)
        except KeyError:
            raise ValueError('%r is not a known password_encoding' % encoding)
        credentials['password'] = cs.decode_password(credentials)
        return credentials