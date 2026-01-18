import base64
import copy
import logging
import sys
import typing
from spnego._context import (
from spnego._credential import (
from spnego._text import to_bytes, to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import GSSError as NativeError
from spnego.exceptions import (
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def _get_gssapi_credential(mech: 'gssapi.OID', usage: str, credentials: typing.List[Credential], context_req: typing.Optional[ContextReq]=None) -> typing.Optional['gssapi.creds.Credentials']:
    """Gets the GSSAPI credential.

    Will get a GSSAPI credential for the mech specified. If the username and password is specified then a new
    set of credentials are explicitly required for the mech specified. Otherwise the credentials are retrieved based on
    the credential type specified.

    Args:
        mech: The mech OID to get the credentials for, only Kerberos is supported.
        usage: Either `initiate` for a client context or `accept` for a server context.
        credentials: List of credentials to retreive from.
        context_req: Context requirement flags that can control how the credential is retrieved.

    Returns:
        gssapi.creds.Credentials: The credential set that was created/retrieved.
    """
    name_type = getattr(gssapi.NameType, 'user' if usage == 'initiate' else 'hostbased_service')
    forwardable = bool(context_req and (context_req & ContextReq.delegate or context_req & ContextReq.delegate_policy))
    for cred in credentials:
        if isinstance(cred, CredentialCache):
            principal = None
            if cred.username:
                principal = gssapi.Name(base=cred.username, name_type=name_type)
            elif usage == 'initiate':
                return None
            gss_cred = gssapi.Credentials(name=principal, usage=usage, mechs=[mech])
            _ = gss_cred.lifetime
            return gss_cred
        elif isinstance(cred, KerberosCCache):
            if usage != 'initiate':
                log.debug('Skipping %s as it can only be used for an initiate Kerberos context', cred)
                continue
            ctx = krb5.init_context()
            ccache = krb5.cc_resolve(ctx, to_bytes(cred.ccache))
            krb5_principal: typing.Optional[krb5.Principal] = None
            if cred.principal:
                krb5_principal = krb5.parse_name_flags(ctx, to_bytes(cred.principal))
            return gssapi.Credentials(base=_gss_acquire_cred_from_ccache(ccache, krb5_principal), usage=usage)
        elif isinstance(cred, (KerberosKeytab, Password)):
            if usage != 'initiate':
                log.debug('Skipping %s as it can only be used for an initiate Kerberos context', cred)
                continue
            if isinstance(cred, KerberosKeytab):
                username = cred.principal or ''
                password = cred.keytab
                is_keytab = True
            else:
                username = cred.username
                password = cred.password
                is_keytab = False
            raw_cred = _kinit(to_bytes(username), to_bytes(password), forwardable=forwardable, is_keytab=is_keytab)
            return gssapi.Credentials(base=raw_cred, usage=usage)
        else:
            log.debug('Skipping credential %s as it does not support required mech type', cred)
            continue
    raise InvalidCredentialError(context_msg='No applicable credentials available')