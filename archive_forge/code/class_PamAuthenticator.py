import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
class PamAuthenticator:
    code = 0
    reason = None

    def __init__(self):
        libc = cdll.LoadLibrary(None)
        self.libc = libc
        libpam = CDLL(find_library('pam'))
        libpam_misc = CDLL(find_library('pam_misc'))
        self.handle = None
        self.messages = []
        self.calloc = libc.calloc
        self.calloc.restype = c_void_p
        self.calloc.argtypes = [c_size_t, c_size_t]
        if hasattr(libpam, 'pam_end'):
            self.pam_end = libpam.pam_end
            self.pam_end.restype = c_int
            self.pam_end.argtypes = [PamHandle, c_int]
        self.pam_start = libpam.pam_start
        self.pam_start.restype = c_int
        self.pam_start.argtypes = [c_char_p, c_char_p, POINTER(PamConv), POINTER(PamHandle)]
        self.pam_acct_mgmt = libpam.pam_acct_mgmt
        self.pam_acct_mgmt.restype = c_int
        self.pam_acct_mgmt.argtypes = [PamHandle, c_int]
        self.pam_set_item = libpam.pam_set_item
        self.pam_set_item.restype = c_int
        self.pam_set_item.argtypes = [PamHandle, c_int, c_void_p]
        self.pam_setcred = libpam.pam_setcred
        self.pam_strerror = libpam.pam_strerror
        self.pam_strerror.restype = c_char_p
        self.pam_strerror.argtypes = [PamHandle, c_int]
        self.pam_authenticate = libpam.pam_authenticate
        self.pam_authenticate.restype = c_int
        self.pam_authenticate.argtypes = [PamHandle, c_int]
        self.pam_open_session = libpam.pam_open_session
        self.pam_open_session.restype = c_int
        self.pam_open_session.argtypes = [PamHandle, c_int]
        self.pam_close_session = libpam.pam_close_session
        self.pam_close_session.restype = c_int
        self.pam_close_session.argtypes = [PamHandle, c_int]
        self.pam_putenv = libpam.pam_putenv
        self.pam_putenv.restype = c_int
        self.pam_putenv.argtypes = [PamHandle, c_char_p]
        if libpam_misc._name:
            self.pam_misc_setenv = libpam_misc.pam_misc_setenv
            self.pam_misc_setenv.restype = c_int
            self.pam_misc_setenv.argtypes = [PamHandle, c_char_p, c_char_p, c_int]
        self.pam_getenv = libpam.pam_getenv
        self.pam_getenv.restype = c_char_p
        self.pam_getenv.argtypes = [PamHandle, c_char_p]
        self.pam_getenvlist = libpam.pam_getenvlist
        self.pam_getenvlist.restype = POINTER(c_char_p)
        self.pam_getenvlist.argtypes = [PamHandle]

    def authenticate(self, username, password, service='login', env=None, call_end=True, encoding='utf-8', resetcreds=True, print_failure_messages=False):
        """username and password authentication for the given service.

        Returns True for success, or False for failure.

        self.code (integer) and self.reason (string) are always stored and may
        be referenced for the reason why authentication failed. 0/'Success'
        will be stored for success.

        Python3 expects bytes() for ctypes inputs.  This function will make
        necessary conversions using the supplied encoding.

        Args:
          username (str): username to authenticate
          password (str): password in plain text
          service (str):  PAM service to authenticate against, defaults to 'login'
          env (dict):      Pam environment variables
          call_end (bool): call the pam_end() function after (default true)
          print_failure_messages (bool): Print messages on failure

        Returns:
          success:  PAM_SUCCESS
          failure:  False
        """

        @conv_func
        def __conv(n_messages, messages, p_response, app_data):
            pyob = cast(app_data, py_object).value
            msg_list = pyob.get('msgs')
            password = pyob.get('password')
            encoding = pyob.get('encoding')
            return my_conv(n_messages, messages, p_response, self.libc, msg_list, password, encoding)
        if isinstance(username, six.text_type):
            username = username.encode(encoding)
        if isinstance(password, six.text_type):
            password = password.encode(encoding)
        if isinstance(service, six.text_type):
            service = service.encode(encoding)
        if b'\x00' in username or b'\x00' in password or b'\x00' in service:
            self.code = PAM_SYSTEM_ERR
            self.reason = 'none of username, password, or service may contain NUL'
            raise ValueError(self.reason)
        app_data = {'msgs': self.messages, 'password': password, 'encoding': encoding}
        conv = PamConv(__conv, c_void_p.from_buffer(py_object(app_data)))
        self.handle = PamHandle()
        retval = self.pam_start(service, username, byref(conv), byref(self.handle))
        if retval != PAM_SUCCESS:
            self.code = retval
            self.reason = 'pam_start() failed: %s' % self.pam_strerror(self.handle, retval)
            return False
        ctty = os.environ.get('DISPLAY')
        if not ctty and os.isatty(0):
            ctty = os.ttyname(0)
        if ctty:
            ctty_p = c_char_p(ctty.encode(encoding))
            retval = self.pam_set_item(self.handle, PAM_TTY, ctty_p)
            retval = self.pam_set_item(self.handle, PAM_XDISPLAY, ctty_p)
        if env:
            if not isinstance(env, dict):
                raise TypeError('"env" must be a dict')
            for key, value in env.items():
                if isinstance(key, bytes) and b'\x00' in key:
                    raise ValueError('"env{}" key cannot contain NULLs')
                if isinstance(value, bytes) and b'\x00' in value:
                    raise ValueError('"env{}" value cannot contain NULLs')
                name_value = '{}={}'.format(key, value)
                retval = self.putenv(name_value, encoding)
        auth_success = self.pam_authenticate(self.handle, 0)
        if auth_success == PAM_SUCCESS:
            auth_success = self.pam_acct_mgmt(self.handle, 0)
        if auth_success == PAM_SUCCESS and resetcreds:
            auth_success = self.pam_setcred(self.handle, PAM_REINITIALIZE_CRED)
        self.code = auth_success
        self.reason = self.pam_strerror(self.handle, auth_success)
        if sys.version_info >= (3,):
            self.reason = self.reason.decode(encoding)
        if call_end and hasattr(self, 'pam_end'):
            self.pam_end(self.handle, auth_success)
            self.handle = None
        if print_failure_messages and self.code != PAM_SUCCESS:
            print(f'Failure: {self.reason}')
        return auth_success == PAM_SUCCESS

    def end(self):
        """A direct call to pam_end()
        Returns:
          Linux-PAM return value as int
        """
        if not self.handle or not hasattr(self, 'pam_end'):
            return PAM_SYSTEM_ERR
        retval = self.pam_end(self.handle, self.code)
        self.handle = None
        return retval

    def open_session(self, encoding='utf-8'):
        """Call pam_open_session as required by the pam_api
        Returns:
          Linux-PAM return value as int
        """
        if not self.handle:
            return PAM_SYSTEM_ERR
        retval = self.pam_open_session(self.handle, 0)
        self.code = retval
        self.reason = self.pam_strerror(self.handle, retval)
        if sys.version_info >= (3,):
            self.reason = self.reason.decode(encoding)
        return retval

    def close_session(self, encoding='utf-8'):
        """Call pam_close_session as required by the pam_api
        Returns:
          Linux-PAM return value as int
        """
        if not self.handle:
            return PAM_SYSTEM_ERR
        retval = self.pam_close_session(self.handle, 0)
        self.code = retval
        self.reason = self.pam_strerror(self.handle, retval)
        if sys.version_info >= (3,):
            self.reason = self.reason.decode(encoding)
        return retval

    def misc_setenv(self, name, value, readonly, encoding='utf-8'):
        """A wrapper for the pam_misc_setenv function
        Args:
          name: key name of the environment variable
          value: the value of the environment variable
        Returns:
          Linux-PAM return value as int
        """
        if not self.handle or not hasattr(self, 'pam_misc_setenv'):
            return PAM_SYSTEM_ERR
        return self.pam_misc_setenv(self.handle, name.encode(encoding), value.encode(encoding), readonly)

    def putenv(self, name_value, encoding='utf-8'):
        """A wrapper for the pam_putenv function
        Args:
          name_value: environment variable in the format KEY=VALUE
                      Without an '=' delete the corresponding variable
        Returns:
          Linux-PAM return value as int
        """
        if not self.handle:
            return PAM_SYSTEM_ERR
        name_value = name_value.encode(encoding)
        retval = self.pam_putenv(self.handle, name_value)
        if retval != PAM_SUCCESS:
            raise Exception(self.pam_strerror(self.handle, retval))
        return retval

    def getenv(self, key, encoding='utf-8'):
        """A wrapper for the pam_getenv function
        Args:
          key name of the environment variable
        Returns:
          value of the environment variable or None on error
        """
        if not self.handle:
            return PAM_SYSTEM_ERR
        if sys.version_info >= (3,):
            if isinstance(key, six.text_type):
                key = key.encode(encoding)
        value = self.pam_getenv(self.handle, key)
        if isinstance(value, type(None)):
            return
        if isinstance(value, int):
            raise Exception(self.pam_strerror(self.handle, value))
        if sys.version_info >= (3,):
            value = value.decode(encoding)
        return value

    def getenvlist(self, encoding='utf-8'):
        """A wrapper for the pam_getenvlist function
        Returns:
          environment as python dictionary
        """
        if not self.handle:
            return PAM_SYSTEM_ERR
        env_list = self.pam_getenvlist(self.handle)
        env_count = 0
        pam_env_items = {}
        while True:
            try:
                item = env_list[env_count]
            except IndexError:
                break
            if not item:
                break
            env_item = item
            if sys.version_info >= (3,):
                env_item = env_item.decode(encoding)
            try:
                pam_key, pam_value = env_item.split('=', 1)
            except ValueError:
                pass
            else:
                pam_env_items[pam_key] = pam_value
            env_count += 1
        return pam_env_items