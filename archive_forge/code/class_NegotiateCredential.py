import ssl
import struct
import typing as t
from ._certificate import get_tls_server_end_point_data
from .client import Credential, MessageEncryptor, SyncLDAPClient
class NegotiateCredential(Credential):

    def __init__(self, username: t.Optional[str]=None, password: t.Optional[str]=None, protocol: str='negotiate', encrypt: bool=True) -> None:
        if SPNEGO_IMPORT_ERR:
            raise ImportError(str(SPNEGO_IMPORT_ERR)) from SPNEGO_IMPORT_ERR
        self.username = username
        self.password = password
        self.protocol = protocol
        self.encrypt = encrypt

    def authenticate(self, client: SyncLDAPClient, *, tls_sock: t.Optional[ssl.SSLSocket]=None) -> None:
        context_req = spnego.ContextReq.mutual_auth
        if tls_sock or not self.encrypt:
            needs_encryptor = False
            context_req |= spnego.ContextReq.no_integrity
        else:
            needs_encryptor = True
            context_req |= spnego.ContextReq.sequence_detect | spnego.ContextReq.integrity | spnego.ContextReq.confidentiality
        ctx = spnego.client(username=self.username, password=self.password, hostname=client.server, service='ldap', context_req=context_req)
        cbt = None
        if tls_sock:
            app_data = get_tls_server_end_point_data(tls_sock.getpeercert(True))
            if app_data:
                cbt = spnego.channel_bindings.GssChannelBindings(application_data=app_data)
        in_token: t.Optional[bytes] = None
        while not ctx.complete:
            out_token = ctx.step(in_token=in_token, channel_bindings=cbt)
            if not out_token:
                break
            in_token = client.bind('', sansldap.SaslCredential('GSS-SPNEGO', out_token), success_only=ctx.complete)
        if needs_encryptor:
            client.register_encryptor(SpnegoEncryptor(ctx))