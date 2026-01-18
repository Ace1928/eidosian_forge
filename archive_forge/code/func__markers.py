import base64
from rsa._compat import is_bytes, range
def _markers(pem_marker):
    """
    Returns the start and end PEM markers, as bytes.
    """
    if not is_bytes(pem_marker):
        pem_marker = pem_marker.encode('ascii')
    return (b'-----BEGIN ' + pem_marker + b'-----', b'-----END ' + pem_marker + b'-----')