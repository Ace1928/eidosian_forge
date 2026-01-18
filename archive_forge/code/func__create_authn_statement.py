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
def _create_authn_statement(self, issuer, expiration_time):
    """Create an object that represents a SAML AuthnStatement.

        <ns0:AuthnStatement xmlns:ns0="urn:oasis:names:tc:SAML:2.0:assertion"
          AuthnInstant="2014-07-30T03:04:25Z" SessionIndex="47335964efb"
          SessionNotOnOrAfter="2014-07-30T03:04:26Z">
            <ns0:AuthnContext>
                <ns0:AuthnContextClassRef>
                  urn:oasis:names:tc:SAML:2.0:ac:classes:Password
                </ns0:AuthnContextClassRef>
                <ns0:AuthenticatingAuthority>
                  https://acme.com/FIM/sps/openstack/saml20
                </ns0:AuthenticatingAuthority>
            </ns0:AuthnContext>
        </ns0:AuthnStatement>

        :returns: XML <AuthnStatement> object

        """
    authn_statement = saml.AuthnStatement()
    authn_statement.authn_instant = utils.isotime()
    authn_statement.session_index = uuid.uuid4().hex
    authn_statement.session_not_on_or_after = expiration_time
    authn_context = saml.AuthnContext()
    authn_context_class = saml.AuthnContextClassRef()
    authn_context_class.set_text(saml.AUTHN_PASSWORD)
    authn_authority = saml.AuthenticatingAuthority()
    authn_authority.set_text(issuer)
    authn_context.authn_context_class_ref = authn_context_class
    authn_context.authenticating_authority = authn_authority
    authn_statement.authn_context = authn_context
    return authn_statement