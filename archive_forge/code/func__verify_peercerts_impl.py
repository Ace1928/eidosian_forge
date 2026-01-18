import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
def _verify_peercerts_impl(ssl_context: ssl.SSLContext, cert_chain: list[bytes], server_hostname: str | None=None) -> None:
    """Verify the cert_chain from the server using Windows APIs."""
    pCertContext = None
    hIntermediateCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        for cert_bytes in cert_chain[1:]:
            CertAddEncodedCertificateToStore(hIntermediateCertStore, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, cert_bytes, len(cert_bytes), CERT_STORE_ADD_USE_EXISTING, None)
        leaf_cert = cert_chain[0]
        pCertContext = CertCreateCertificateContext(X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, leaf_cert, len(leaf_cert))
        cert_enhkey_usage = CERT_ENHKEY_USAGE()
        cert_enhkey_usage.cUsageIdentifier = 1
        cert_enhkey_usage.rgpszUsageIdentifier = (c_char_p * 1)(OID_PKIX_KP_SERVER_AUTH)
        cert_usage_match = CERT_USAGE_MATCH()
        cert_usage_match.Usage = cert_enhkey_usage
        chain_params = CERT_CHAIN_PARA()
        chain_params.RequestedUsage = cert_usage_match
        chain_params.cbSize = sizeof(chain_params)
        pChainPara = pointer(chain_params)
        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_CHAIN
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_END_CERT
        else:
            chain_flags = 0
        try:
            _get_and_verify_cert_chain(ssl_context, None, hIntermediateCertStore, pCertContext, pChainPara, server_hostname, chain_flags=chain_flags)
        except ssl.SSLCertVerificationError:
            custom_ca_certs: list[bytes] | None = ssl_context.get_ca_certs(binary_form=True)
            if custom_ca_certs:
                _verify_using_custom_ca_certs(ssl_context, custom_ca_certs, hIntermediateCertStore, pCertContext, pChainPara, server_hostname, chain_flags=chain_flags)
            else:
                raise
    finally:
        CertCloseStore(hIntermediateCertStore, 0)
        if pCertContext:
            CertFreeCertificateContext(pCertContext)