from __future__ import unicode_literals
import sys
import datetime
from decimal import Decimal
import os
import logging
import hashlib
import warnings
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement
import random
import string
from hashlib import sha1
class BinaryTokenSignature:
    """WebService Security extension to add a basic signature to xml request"""

    def __init__(self, certificate='', private_key='', password=None, cacert=None):
        self.certificate = ''.join([line for line in open(certificate) if not line.startswith('---')])
        self.private_key = private_key
        self.password = password
        self.cacert = cacert

    def preprocess(self, client, request, method, args, kwargs, headers, soap_uri):
        """Sign the outgoing SOAP request"""
        body = request('Body', ns=soap_uri)
        header = request('Header', ns=soap_uri)
        body['wsu:Id'] = 'id-14'
        body['xmlns:wsu'] = WSU_URI
        for attr, value in request[:]:
            if attr.startswith('xmlns'):
                body[attr] = value
        ref_xml = repr(body)
        from . import xmlsec
        vars = xmlsec.rsa_sign(ref_xml, '#id-14', self.private_key, self.password)
        vars['certificate'] = self.certificate
        wsse = SimpleXMLElement(BIN_TOKEN_TMPL % vars)
        header.import_node(wsse)

    def postprocess(self, client, response, method, args, kwargs, headers, soap_uri):
        """Verify the signature of the incoming response"""
        from . import xmlsec
        body = response('Body', ns=soap_uri)
        header = response('Header', ns=soap_uri)
        wsse = header('Security', ns=WSSE_URI)
        cert = wsse('BinarySecurityToken', ns=WSSE_URI)
        self.__check(cert['EncodingType'], Base64Binary_URI)
        self.__check(cert['ValueType'], X509v3_URI)
        cert_der = str(cert).decode('base64')
        public_key = xmlsec.x509_extract_rsa_public_key(cert_der, binary=True)
        if not self.cacert:
            warnings.warn('No CA provided, WSSE not validating certificate')
        elif not xmlsec.x509_verify(self.cacert, cert_der, binary=True):
            raise RuntimeError('WSSE certificate validation failed')
        self.__check(body['xmlns:wsu'], WSU_URI)
        ref_uri = body['wsu:Id']
        signature = wsse('Signature', ns=XMLDSIG_URI)
        signed_info = signature('SignedInfo', ns=XMLDSIG_URI)
        signature_value = signature('SignatureValue', ns=XMLDSIG_URI)
        self.__check(signed_info('Reference', ns=XMLDSIG_URI)['URI'], '#' + ref_uri)
        self.__check(signed_info('SignatureMethod', ns=XMLDSIG_URI)['Algorithm'], XMLDSIG_URI + 'rsa-sha1')
        self.__check(signed_info('Reference', ns=XMLDSIG_URI)('DigestMethod', ns=XMLDSIG_URI)['Algorithm'], XMLDSIG_URI + 'sha1')
        for attr, value in response[:]:
            if attr.startswith('xmlns'):
                body[attr] = value
        ref_xml = xmlsec.canonicalize(repr(body))
        computed_hash = xmlsec.sha1_hash_digest(ref_xml)
        digest_value = str(signed_info('Reference', ns=XMLDSIG_URI)('DigestValue', ns=XMLDSIG_URI))
        if computed_hash != digest_value:
            raise RuntimeError('WSSE SHA1 hash digests mismatch')
        signed_info['xmlns'] = XMLDSIG_URI
        xml = repr(signed_info)
        ok = xmlsec.rsa_verify(xml, str(signature_value), public_key)
        if not ok:
            raise RuntimeError('WSSE RSA-SHA1 signature verification failed')

    def __check(self, value, expected, msg='WSSE sanity check failed'):
        if value != expected:
            raise RuntimeError(msg)