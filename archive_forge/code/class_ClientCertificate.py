import ssl
import struct
import typing as t
from ._certificate import get_tls_server_end_point_data
from .client import Credential, MessageEncryptor, SyncLDAPClient
class ClientCertificate(Credential):

    def authenticate(self, client: SyncLDAPClient, *, tls_sock: t.Optional[ssl.SSLSocket]=None) -> None:
        client.bind('', sansldap.SaslCredential('EXTERNAL', b''))