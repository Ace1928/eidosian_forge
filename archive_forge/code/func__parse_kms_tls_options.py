from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def _parse_kms_tls_options(kms_tls_options: Optional[Mapping[str, Any]]) -> dict[str, SSLContext]:
    """Parse KMS TLS connection options."""
    if not kms_tls_options:
        return {}
    if not isinstance(kms_tls_options, dict):
        raise TypeError('kms_tls_options must be a dict')
    contexts = {}
    for provider, options in kms_tls_options.items():
        if not isinstance(options, dict):
            raise TypeError(f'kms_tls_options["{provider}"] must be a dict')
        options.setdefault('tls', True)
        opts = _CaseInsensitiveDictionary(options)
        opts = _handle_security_options(opts)
        opts = _normalize_options(opts)
        opts = cast(_CaseInsensitiveDictionary, validate_options(opts))
        ssl_context, allow_invalid_hostnames = _parse_ssl_options(opts)
        if ssl_context is None:
            raise ConfigurationError('TLS is required for KMS providers')
        if allow_invalid_hostnames:
            raise ConfigurationError('Insecure TLS options prohibited')
        for n in ['tlsInsecure', 'tlsAllowInvalidCertificates', 'tlsAllowInvalidHostnames', 'tlsDisableCertificateRevocationCheck']:
            if n in opts:
                raise ConfigurationError(f'Insecure TLS options prohibited: {n}')
            contexts[provider] = ssl_context
    return contexts