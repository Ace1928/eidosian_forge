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
def _create_subject(self, user, expiration_time, recipient):
    """Create an object that represents a SAML Subject.

        <ns0:Subject>
            <ns0:NameID>
                john@smith.com</ns0:NameID>
            <ns0:SubjectConfirmation
              Method="urn:oasis:names:tc:SAML:2.0:cm:bearer">
                <ns0:SubjectConfirmationData
                  NotOnOrAfter="2014-08-19T11:53:57.243106Z"
                  Recipient="http://beta.com/Shibboleth.sso/SAML2/POST" />
            </ns0:SubjectConfirmation>
        </ns0:Subject>

        :returns: XML <Subject> object

        """
    name_id = saml.NameID()
    name_id.set_text(user)
    subject_conf_data = saml.SubjectConfirmationData()
    subject_conf_data.recipient = recipient
    subject_conf_data.not_on_or_after = expiration_time
    subject_conf = saml.SubjectConfirmation()
    subject_conf.method = saml.SCM_BEARER
    subject_conf.subject_confirmation_data = subject_conf_data
    subject = saml.Subject()
    subject.subject_confirmation = subject_conf
    subject.name_id = name_id
    return subject