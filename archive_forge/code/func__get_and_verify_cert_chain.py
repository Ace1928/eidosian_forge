import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
def _get_and_verify_cert_chain(ssl_context: ssl.SSLContext, hChainEngine: HCERTCHAINENGINE | None, hIntermediateCertStore: HCERTSTORE, pPeerCertContext: c_void_p, pChainPara: PCERT_CHAIN_PARA, server_hostname: str | None, chain_flags: int) -> None:
    ppChainContext = None
    try:
        ppChainContext = pointer(PCERT_CHAIN_CONTEXT())
        CertGetCertificateChain(hChainEngine, pPeerCertContext, None, hIntermediateCertStore, pChainPara, chain_flags, None, ppChainContext)
        pChainContext = ppChainContext.contents
        ssl_extra_cert_chain_policy_para = SSL_EXTRA_CERT_CHAIN_POLICY_PARA()
        ssl_extra_cert_chain_policy_para.cbSize = sizeof(ssl_extra_cert_chain_policy_para)
        ssl_extra_cert_chain_policy_para.dwAuthType = AUTHTYPE_SERVER
        ssl_extra_cert_chain_policy_para.fdwChecks = 0
        if server_hostname:
            ssl_extra_cert_chain_policy_para.pwszServerName = c_wchar_p(server_hostname)
        chain_policy = CERT_CHAIN_POLICY_PARA()
        chain_policy.pvExtraPolicyPara = cast(pointer(ssl_extra_cert_chain_policy_para), c_void_p)
        if ssl_context.verify_mode == ssl.CERT_NONE:
            chain_policy.dwFlags |= CERT_CHAIN_POLICY_VERIFY_MODE_NONE_FLAGS
        if not ssl_context.check_hostname:
            chain_policy.dwFlags |= CERT_CHAIN_POLICY_IGNORE_INVALID_NAME_FLAG
        chain_policy.cbSize = sizeof(chain_policy)
        pPolicyPara = pointer(chain_policy)
        policy_status = CERT_CHAIN_POLICY_STATUS()
        policy_status.cbSize = sizeof(policy_status)
        pPolicyStatus = pointer(policy_status)
        CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, pChainContext, pPolicyPara, pPolicyStatus)
        error_code = policy_status.dwError
        if error_code:
            error_message_buf = create_unicode_buffer(1024)
            error_message_chars = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, None, error_code, 0, error_message_buf, sizeof(error_message_buf), None)
            if error_message_chars <= 0:
                error_message = f'Certificate chain policy error {error_code:#x} [{policy_status.lElementIndex}]'
            else:
                error_message = error_message_buf.value.strip()
            err = ssl.SSLCertVerificationError(error_message)
            err.verify_message = error_message
            err.verify_code = error_code
            raise err from None
    finally:
        if ppChainContext:
            CertFreeCertificateChain(ppChainContext.contents)