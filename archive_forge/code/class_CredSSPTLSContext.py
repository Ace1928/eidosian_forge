import dataclasses
import datetime
import platform
import ssl
import typing
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
@dataclasses.dataclass
class CredSSPTLSContext:
    """A TLS context generated for CredSSP.

    This is the SSLContext object used by both an initiator and acceptor of
    CredSSP authentication. It allows the caller to finely control the SSL/TLS
    options used during the CredSSP authentication phase, e.g. selecting
    min/max protocol versions, specific cipher suites, etc.

    The public_key attribute is used for acceptor CredSSP contexts as the
    DER encoded public key loaded in the SSLContext. Here is an example of
    generating and loading a X509 certificate for an acceptor context:

        ctx = spnego.tls.default_tls_context()
        cert_pem, key_Pem, pub_key = spnego.tls.generate_tls_certificate()

        # Cannot use tempfile.NamedTemporaryFile due to sharing violations on
        # Windows. Use a tempdir as a workaround.
        temp_dir = tempfile.mkdtemp()
        try:
            cert_path = os.path.join(tmpe_dir, 'ca.pem')
            with open(cert_path, mode'wb') as fd:
                fd.write(cert_pem)
                fd.write(key_pem)

            ctx.context.load_cert_chain(cert_path)
            ctx.public_key = pub_key

        finally:
            shutil.rmtree(temp_dir)

    This context is then passed in through the `credssp_tls_context` kwarg of
    :meth:`spnego.client` or :meth:`spnego.server`.

    Attributes:
        context (ssl.SSLContext): The TLS context generated for CredSSP.
        public_key (Optional[bytes]): When generating the TLS context for an
            acceptor this is the public key bytes for the generated cert in the
            TLS context.
    """
    context: ssl.SSLContext
    public_key: typing.Optional[bytes] = None