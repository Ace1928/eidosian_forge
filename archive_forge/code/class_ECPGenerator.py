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
class ECPGenerator(object):
    """A class for generating an ECP assertion."""

    @staticmethod
    def generate_ecp(saml_assertion, relay_state_prefix):
        ecp_generator = ECPGenerator()
        header = ecp_generator._create_header(relay_state_prefix)
        body = ecp_generator._create_body(saml_assertion)
        envelope = soapenv.Envelope(header=header, body=body)
        return envelope

    def _create_header(self, relay_state_prefix):
        relay_state_text = relay_state_prefix + uuid.uuid4().hex
        relay_state = ecp.RelayState(actor=client_base.ACTOR, must_understand='1', text=relay_state_text)
        header = soapenv.Header()
        header.extension_elements = [saml2.element_to_extension_element(relay_state)]
        return header

    def _create_body(self, saml_assertion):
        body = soapenv.Body()
        body.extension_elements = [saml2.element_to_extension_element(saml_assertion)]
        return body