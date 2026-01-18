from __future__ import annotations
import email.base64mime
import email.generator
import email.message
import email.policy
import io
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import pkcs7 as rust_pkcs7
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.utils import _check_byteslike
class PKCS7Options(utils.Enum):
    Text = 'Add text/plain MIME type'
    Binary = "Don't translate input data into canonical MIME format"
    DetachedSignature = "Don't embed data in the PKCS7 structure"
    NoCapabilities = "Don't embed SMIME capabilities"
    NoAttributes = "Don't embed authenticatedAttributes"
    NoCerts = "Don't embed signer certificate"