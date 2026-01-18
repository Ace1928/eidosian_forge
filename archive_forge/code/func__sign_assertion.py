import datetime
import os
import subprocess  # nosec : see comments in the code below
import uuid
from oslo_log import log
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import timeutils
import saml2
from saml2 import client_base
from saml2 import md
from saml2.profile import ecp
from saml2 import saml
from saml2 import samlp
from saml2.schema import soapenv
from saml2 import sigver
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _sign_assertion(assertion):
    """Sign a SAML assertion.

    This method utilizes ``xmlsec1`` binary and signs SAML assertions in a
    separate process. ``xmlsec1`` cannot read input data from stdin so the
    prepared assertion needs to be serialized and stored in a temporary file.
    This file will be deleted immediately after ``xmlsec1`` returns. The signed
    assertion is redirected to a standard output and read using
    ``subprocess.PIPE`` redirection. A ``saml.Assertion`` class is created from
    the signed string again and returned.

    Parameters that are required in the CONF::
    * xmlsec_binary
    * private key file path
    * public key file path
    :returns: XML <Assertion> object

    """
    for option in ('keyfile', 'certfile'):
        if ',' in getattr(CONF.saml, option, ''):
            raise exception.UnexpectedError('The configuration value in `keystone.conf [saml] %s` cannot contain a comma (`,`). Please fix your configuration.' % option)
    certificates = '%(idp_private_key)s,%(idp_public_key)s' % {'idp_public_key': CONF.saml.certfile, 'idp_private_key': CONF.saml.keyfile}
    _verify_assertion_binary_is_installed()
    command_list = [CONF.saml.xmlsec1_binary, '--sign', '--privkey-pem', certificates, '--id-attr:ID', 'Assertion']
    file_path = None
    try:
        file_path = fileutils.write_to_tempfile(assertion.to_string(nspair={'saml': saml2.NAMESPACE, 'xmldsig': xmldsig.NAMESPACE}))
        command_list.append(file_path)
        stdout = subprocess.check_output(command_list, stderr=subprocess.STDOUT)
    except Exception as e:
        msg = 'Error when signing assertion, reason: %(reason)s%(output)s'
        LOG.error(msg, {'reason': e, 'output': ' ' + e.output if hasattr(e, 'output') else ''})
        raise exception.SAMLSigningError(reason=e)
    finally:
        try:
            if file_path:
                os.remove(file_path)
        except OSError:
            pass
    return saml2.create_class_from_xml_string(saml.Assertion, stdout)