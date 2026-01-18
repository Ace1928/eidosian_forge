import logging
from .._compat import properties
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError, ExceptionRaisedContext
class DecodingCredential(dict):

    @property
    def value(self):
        """
        Attempt to decode the credential blob as UTF-16 then UTF-8.
        """
        cred = self['CredentialBlob']
        try:
            return cred.decode('utf-16')
        except UnicodeDecodeError:
            decoded_cred_utf8 = cred.decode('utf-8')
            log.warning('Retrieved a UTF-8 encoded credential. Please be aware that this library only writes credentials in UTF-16.')
            return decoded_cred_utf8