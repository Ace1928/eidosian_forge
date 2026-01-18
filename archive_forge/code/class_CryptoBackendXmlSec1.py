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
class CryptoBackendXmlSec1(CryptoBackend):
    """
    CryptoBackend implementation using external binary 1 to sign
    and verify XML documents.
    """
    __DEBUG = 0

    def __init__(self, xmlsec_binary, delete_tmpfiles=True, **kwargs):
        CryptoBackend.__init__(self, **kwargs)
        if not isinstance(xmlsec_binary, str):
            raise ValueError('xmlsec_binary should be of type string')
        self.xmlsec = xmlsec_binary
        self.delete_tmpfiles = delete_tmpfiles
        try:
            self.non_xml_crypto = RSACrypto(kwargs['rsa_key'])
        except KeyError:
            pass

    @property
    def version(self):
        com_list = [self.xmlsec, '--version']
        pof = Popen(com_list, stderr=PIPE, stdout=PIPE)
        content, _ = pof.communicate()
        content = content.decode('ascii')
        try:
            return content.split(' ')[1]
        except IndexError:
            return '0.0.0'

    def encrypt(self, text, recv_key, template, session_key_type, xpath=''):
        """

        :param text: The text to be compiled
        :param recv_key: Filename of a file where the key resides
        :param template: Filename of a file with the pre-encryption part
        :param session_key_type: Type and size of a new session key
            'des-192' generates a new 192 bits DES key for DES3 encryption
        :param xpath: What should be encrypted
        :return:
        """
        logger.debug('Encryption input len: %d', len(text))
        tmp = make_temp(text, decode=False, delete_tmpfiles=self.delete_tmpfiles)
        com_list = [self.xmlsec, '--encrypt', '--pubkey-cert-pem', recv_key, '--session-key', session_key_type, '--xml-data', tmp.name]
        if xpath:
            com_list.extend(['--node-xpath', xpath])
        try:
            _stdout, _stderr, output = self._run_xmlsec(com_list, [template])
        except XmlsecError as e:
            raise EncryptError(com_list) from e
        return output

    def encrypt_assertion(self, statement, enc_key, template, key_type='des-192', node_xpath=None, node_id=None):
        """
        Will encrypt an assertion

        :param statement: A XML document that contains the assertion to encrypt
        :param enc_key: File name of a file containing the encryption key
        :param template: A template for the encryption part to be added.
        :param key_type: The type of session key to use.
        :return: The encrypted text
        """
        if isinstance(statement, SamlBase):
            statement = pre_encrypt_assertion(statement)
        tmp = make_temp(str(statement), decode=False, delete_tmpfiles=self.delete_tmpfiles)
        tmp2 = make_temp(str(template), decode=False, delete_tmpfiles=self.delete_tmpfiles)
        if not node_xpath:
            node_xpath = ASSERT_XPATH
        com_list = [self.xmlsec, '--encrypt', '--pubkey-cert-pem', enc_key, '--session-key', key_type, '--xml-data', tmp.name, '--node-xpath', node_xpath]
        if node_id:
            com_list.extend(['--node-id', node_id])
        try:
            _stdout, _stderr, output = self._run_xmlsec(com_list, [tmp2.name])
        except XmlsecError as e:
            raise EncryptError(com_list) from e
        return output.decode('utf-8')

    def decrypt(self, enctext, key_file):
        """

        :param enctext: XML document containing an encrypted part
        :param key_file: The key to use for the decryption
        :return: The decrypted document
        """
        logger.debug('Decrypt input len: %d', len(enctext))
        tmp = make_temp(enctext, decode=False, delete_tmpfiles=self.delete_tmpfiles)
        com_list = [self.xmlsec, '--decrypt', '--privkey-pem', key_file, '--id-attr:Id', ENC_KEY_CLASS]
        try:
            _stdout, _stderr, output = self._run_xmlsec(com_list, [tmp.name])
        except XmlsecError as e:
            raise DecryptError(com_list) from e
        return output.decode('utf-8')

    def sign_statement(self, statement, node_name, key_file, node_id):
        """
        Sign an XML statement.

        :param statement: The statement to be signed
        :param node_name: string like 'urn:oasis:names:...:Assertion'
        :param key_file: The file where the key can be found
        :param node_id:
        :return: The signed statement
        """
        if isinstance(statement, SamlBase):
            statement = str(statement)
        tmp = make_temp(statement, suffix='.xml', decode=False, delete_tmpfiles=self.delete_tmpfiles)
        com_list = [self.xmlsec, '--sign', '--privkey-pem', key_file, '--id-attr:ID', node_name]
        if node_id:
            com_list.extend(['--node-id', node_id])
        try:
            stdout, stderr, output = self._run_xmlsec(com_list, [tmp.name])
        except XmlsecError as e:
            raise SignatureError(com_list) from e
        if output:
            return output.decode('utf-8')
        if stdout:
            return stdout.decode('utf-8')
        raise SignatureError(stderr)

    def validate_signature(self, signedtext, cert_file, cert_type, node_name, node_id):
        """
        Validate signature on XML document.

        :param signedtext: The XML document as a string
        :param cert_file: The public key that was used to sign the document
        :param cert_type: The file type of the certificate
        :param node_name: The name of the class that is signed
        :param node_id: The identifier of the node
        :return: Boolean True if the signature was correct otherwise False.
        """
        if not isinstance(signedtext, bytes):
            signedtext = signedtext.encode('utf-8')
        tmp = make_temp(signedtext, suffix='.xml', decode=False, delete_tmpfiles=self.delete_tmpfiles)
        com_list = [self.xmlsec, '--verify', '--enabled-reference-uris', 'empty,same-doc', '--enabled-key-data', 'raw-x509-cert', f'--pubkey-cert-{cert_type}', cert_file, '--id-attr:ID', node_name]
        if node_id:
            com_list.extend(['--node-id', node_id])
        try:
            _stdout, stderr, _output = self._run_xmlsec(com_list, [tmp.name])
        except XmlsecError as e:
            raise SignatureError(com_list) from e
        return parse_xmlsec_verify_output(stderr, self.version_nums)

    def _run_xmlsec(self, com_list, extra_args):
        """
        Common code to invoke xmlsec and parse the output.
        :param com_list: Key-value parameter list for xmlsec
        :param extra_args: Positional parameters to be appended after all
            key-value parameters
        :result: Whatever xmlsec wrote to an --output temporary file
        """
        with NamedTemporaryFile(suffix='.xml') as ntf:
            com_list.extend(['--output', ntf.name])
            if self.version_nums >= (1, 3):
                com_list.extend(['--lax-key-search'])
            com_list += extra_args
            logger.debug('xmlsec command: %s', ' '.join(com_list))
            pof = Popen(com_list, stderr=PIPE, stdout=PIPE)
            p_out, p_err = pof.communicate()
            p_out = p_out.decode()
            p_err = p_err.decode()
            if pof.returncode != 0:
                errmsg = f'returncode={pof.returncode}\nerror={p_err}\noutput={p_out}'
                logger.error(errmsg)
                raise XmlsecError(errmsg)
            ntf.seek(0)
            return (p_out, p_err, ntf.read())