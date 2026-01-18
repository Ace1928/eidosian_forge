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
def _smime_encode(data: bytes, signature: bytes, micalg: str, text_mode: bool) -> bytes:
    m = email.message.Message()
    m.add_header('MIME-Version', '1.0')
    m.add_header('Content-Type', 'multipart/signed', protocol='application/x-pkcs7-signature', micalg=micalg)
    m.preamble = 'This is an S/MIME signed message\n'
    msg_part = OpenSSLMimePart()
    msg_part.set_payload(data)
    if text_mode:
        msg_part.add_header('Content-Type', 'text/plain')
    m.attach(msg_part)
    sig_part = email.message.MIMEPart()
    sig_part.add_header('Content-Type', 'application/x-pkcs7-signature', name='smime.p7s')
    sig_part.add_header('Content-Transfer-Encoding', 'base64')
    sig_part.add_header('Content-Disposition', 'attachment', filename='smime.p7s')
    sig_part.set_payload(email.base64mime.body_encode(signature, maxlinelen=65))
    del sig_part['MIME-Version']
    m.attach(sig_part)
    fp = io.BytesIO()
    g = email.generator.BytesGenerator(fp, maxheaderlen=0, mangle_from_=False, policy=m.policy.clone(linesep='\r\n'))
    g.flatten(m)
    return fp.getvalue()