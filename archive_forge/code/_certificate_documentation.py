import os.path
import secrets
import ssl
import tempfile
import typing as t
Loads a certificate to use with client authentication.

    Loads the supplied certificate that can be used for client authentication.
    This function is a wrapper around load_cert_chain and offers the ability to
    load a cert/key from a string or load a PFX formatted certificate with an
    optional password.

    The certificate argument can either be a string of the PEM encoded
    certificate and/or key. It can also be the path to a file of a PEM, DEF, or
    PKCS12 (pfx) certificate and/or key. The key argument can be used to
    specify the certificate key if it is not bundled with the certificate
    argument.

    Args:
        context: The SSLContext to load the cert info.
        certificate: The certificate as a string or filepath.
        key: The optional key as a string or filepath.
        password: The password that is used to decrypt the key or pfx file.
    