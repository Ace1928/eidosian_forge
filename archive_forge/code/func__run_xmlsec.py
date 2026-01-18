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