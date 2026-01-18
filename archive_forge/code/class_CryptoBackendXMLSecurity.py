import base64
import datetime
import hashlib
import itertools
import logging
import os
import re
from subprocess import PIPE
from subprocess import Popen
import sys
from tempfile import NamedTemporaryFile
from time import mktime
from uuid import uuid4 as gen_random_key
import dateutil
from urllib import parse
from OpenSSL import crypto
import pytz
from saml2 import ExtensionElement
from saml2 import SamlBase
from saml2 import SAMLError
from saml2 import class_name
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2 import samlp
from saml2.cert import CertificateError
from saml2.cert import OpenSSLWrapper
from saml2.cert import read_cert_from_file
import saml2.cryptography.asymmetric
import saml2.cryptography.pki
import saml2.data.templates as _data_template
from saml2.extension import pefim
from saml2.extension.pefim import SPCertEnc
from saml2.s_utils import Unsupported
from saml2.saml import EncryptedAssertion
from saml2.time_util import str_to_time
from saml2.xml.schema import XMLSchemaError
from saml2.xml.schema import validate as validate_doc_with_schema
from saml2.xmldsig import ALLOWED_CANONICALIZATIONS
from saml2.xmldsig import ALLOWED_TRANSFORMS
from saml2.xmldsig import SIG_RSA_SHA1
from saml2.xmldsig import SIG_RSA_SHA224
from saml2.xmldsig import SIG_RSA_SHA256
from saml2.xmldsig import SIG_RSA_SHA384
from saml2.xmldsig import SIG_RSA_SHA512
from saml2.xmldsig import TRANSFORM_C14N
from saml2.xmldsig import TRANSFORM_ENVELOPED
import saml2.xmldsig as ds
from saml2.xmlenc import CipherData
from saml2.xmlenc import CipherValue
from saml2.xmlenc import EncryptedData
from saml2.xmlenc import EncryptedKey
from saml2.xmlenc import EncryptionMethod
class CryptoBackendXMLSecurity(CryptoBackend):
    """
    CryptoBackend implementation using pyXMLSecurity to sign and verify
    XML documents.

    Encrypt and decrypt is currently unsupported by pyXMLSecurity.

    pyXMLSecurity uses lxml (libxml2) to parse XML data, but otherwise
    try to get by with native Python code. It does native Python RSA
    signatures, or alternatively PyKCS11 to offload cryptographic work
    to an external PKCS#11 module.
    """

    def __init__(self):
        CryptoBackend.__init__(self)

    @property
    def version(self):
        try:
            import xmlsec
            return xmlsec.__version__
        except (ImportError, AttributeError):
            return '0.0.0'

    def sign_statement(self, statement, node_name, key_file, node_id):
        """
        Sign an XML statement.

        The parameters actually used in this CryptoBackend
        implementation are :

        :param statement: XML as string
        :param node_name: Name of the node to sign
        :param key_file: xmlsec key_spec string(), filename,
            'pkcs11://' URI or PEM data
        :returns: Signed XML as string
        """
        import lxml.etree
        import xmlsec
        xml = xmlsec.parse_xml(statement)
        signed = xmlsec.sign(xml, key_file)
        signed_str = lxml.etree.tostring(signed, xml_declaration=False, encoding='UTF-8')
        if not isinstance(signed_str, str):
            signed_str = signed_str.decode('utf-8')
        return signed_str

    def validate_signature(self, signedtext, cert_file, cert_type, node_name, node_id):
        """
        Validate signature on XML document.

        The parameters actually used in this CryptoBackend
        implementation are :

        :param signedtext: The signed XML data as string
        :param cert_file: xmlsec key_spec string(), filename,
            'pkcs11://' URI or PEM data
        :param cert_type: string, must be 'pem' for now
        :returns: True on successful validation, False otherwise
        """
        if cert_type != 'pem':
            raise Unsupported('Only PEM certs supported here')
        import xmlsec
        xml = xmlsec.parse_xml(signedtext)
        try:
            return xmlsec.verify(xml, cert_file)
        except xmlsec.XMLSigException:
            return False