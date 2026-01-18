import logging
from typing import Any, Dict
from OpenSSL import SSL
from service_identity.exceptions import CertificateError
from twisted.internet._sslverify import (
from twisted.internet.ssl import AcceptableCiphers
from scrapy.utils.ssl import get_temp_key_info, x509name_to_string
class ScrapyClientTLSOptions(ClientTLSOptions):
    """
    SSL Client connection creator ignoring certificate verification errors
    (for genuinely invalid certificates or bugs in verification code).

    Same as Twisted's private _sslverify.ClientTLSOptions,
    except that VerificationError, CertificateError and ValueError
    exceptions are caught, so that the connection is not closed, only
    logging warnings. Also, HTTPS connection parameters logging is added.
    """

    def __init__(self, hostname: str, ctx: SSL.Context, verbose_logging: bool=False):
        super().__init__(hostname, ctx)
        self.verbose_logging: bool = verbose_logging

    def _identityVerifyingInfoCallback(self, connection: SSL.Connection, where: int, ret: Any) -> None:
        if where & SSL.SSL_CB_HANDSHAKE_START:
            connection.set_tlsext_host_name(self._hostnameBytes)
        elif where & SSL.SSL_CB_HANDSHAKE_DONE:
            if self.verbose_logging:
                logger.debug('SSL connection to %s using protocol %s, cipher %s', self._hostnameASCII, connection.get_protocol_version_name(), connection.get_cipher_name())
                server_cert = connection.get_peer_certificate()
                if server_cert:
                    logger.debug('SSL connection certificate: issuer "%s", subject "%s"', x509name_to_string(server_cert.get_issuer()), x509name_to_string(server_cert.get_subject()))
                key_info = get_temp_key_info(connection._ssl)
                if key_info:
                    logger.debug('SSL temp key: %s', key_info)
            try:
                verifyHostname(connection, self._hostnameASCII)
            except (CertificateError, VerificationError) as e:
                logger.warning('Remote certificate is not valid for hostname "%s"; %s', self._hostnameASCII, e)
            except ValueError as e:
                logger.warning('Ignoring error while verifying certificate from host "%s" (exception: %r)', self._hostnameASCII, e)