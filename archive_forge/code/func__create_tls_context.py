import base64
import hashlib
import logging
import os
import re
import shutil
import ssl
import struct
import tempfile
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, Password, unify_credentials
from spnego._credssp_structures import (
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.tls import (
def _create_tls_context(usage: str) -> CredSSPTLSContext:
    log.debug('Creating TLS context')
    ctx = default_tls_context(usage=usage)
    if usage == 'accept':
        global _X509_CERTIFICATE
        if not _X509_CERTIFICATE:
            _X509_CERTIFICATE = generate_tls_certificate()
        cert_pem, key_pem, public_key = _X509_CERTIFICATE
        temp_dir = tempfile.mkdtemp()
        try:
            cert_path = os.path.join(temp_dir, 'ca.pem')
            with open(cert_path, mode='wb') as fd:
                fd.write(cert_pem)
                fd.write(key_pem)
            ctx.context.load_cert_chain(cert_path)
            ctx.public_key = public_key
        finally:
            shutil.rmtree(temp_dir)
    return ctx