from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, cast
from bson.codec_options import _parse_codec_options
from pymongo import common
from pymongo.auth import MongoCredential, _build_credentials_tuple
from pymongo.common import validate_boolean
from pymongo.compression_support import CompressionSettings
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _EventListener, _EventListeners
from pymongo.pool import PoolOptions
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import (
from pymongo.server_selectors import any_server_selector
from pymongo.ssl_support import get_ssl_context
from pymongo.write_concern import WriteConcern
def _parse_ssl_options(options: Mapping[str, Any]) -> tuple[Optional[SSLContext], bool]:
    """Parse ssl options."""
    use_tls = options.get('tls')
    if use_tls is not None:
        validate_boolean('tls', use_tls)
    certfile = options.get('tlscertificatekeyfile')
    passphrase = options.get('tlscertificatekeyfilepassword')
    ca_certs = options.get('tlscafile')
    crlfile = options.get('tlscrlfile')
    allow_invalid_certificates = options.get('tlsallowinvalidcertificates', False)
    allow_invalid_hostnames = options.get('tlsallowinvalidhostnames', False)
    disable_ocsp_endpoint_check = options.get('tlsdisableocspendpointcheck', False)
    enabled_tls_opts = []
    for opt in ('tlscertificatekeyfile', 'tlscertificatekeyfilepassword', 'tlscafile', 'tlscrlfile'):
        if opt in options and options[opt]:
            enabled_tls_opts.append(opt)
    for opt in ('tlsallowinvalidcertificates', 'tlsallowinvalidhostnames', 'tlsdisableocspendpointcheck'):
        if opt in options and (not options[opt]):
            enabled_tls_opts.append(opt)
    if enabled_tls_opts:
        if use_tls is None:
            use_tls = True
        elif not use_tls:
            raise ConfigurationError('TLS has not been enabled but the following tls parameters have been set: %s. Please set `tls=True` or remove.' % ', '.join(enabled_tls_opts))
    if use_tls:
        ctx = get_ssl_context(certfile, passphrase, ca_certs, crlfile, allow_invalid_certificates, allow_invalid_hostnames, disable_ocsp_endpoint_check)
        return (ctx, allow_invalid_hostnames)
    return (None, allow_invalid_hostnames)