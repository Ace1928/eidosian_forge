import os.path
import secrets
import ssl
import tempfile
import typing as t
def _try_load_der_cert(data: bytes) -> t.Optional[str]:
    if not HAS_CRYPTOGRAPHY:
        return None
    try:
        cert = x509.load_der_x509_certificate(data)
    except ValueError:
        return None
    else:
        return cert.public_bytes(encoding=serialization.Encoding.PEM).decode()