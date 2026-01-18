import calendar
import logging
from saml2 import SAMLError
from saml2 import class_name
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2 import samlp
from saml2 import time_util
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.attribute_converter import to_local
from saml2.s_utils import RequestVersionTooHigh
from saml2.s_utils import RequestVersionTooLow
from saml2.saml import SCM_BEARER
from saml2.saml import SCM_HOLDER_OF_KEY
from saml2.saml import SCM_SENDER_VOUCHES
from saml2.saml import XSI_TYPE
from saml2.saml import attribute_from_string
from saml2.saml import encrypted_attribute_from_string
from saml2.samlp import STATUS_AUTHN_FAILED
from saml2.samlp import STATUS_INVALID_ATTR_NAME_OR_VALUE
from saml2.samlp import STATUS_INVALID_NAMEID_POLICY
from saml2.samlp import STATUS_NO_AUTHN_CONTEXT
from saml2.samlp import STATUS_NO_AVAILABLE_IDP
from saml2.samlp import STATUS_NO_PASSIVE
from saml2.samlp import STATUS_NO_SUPPORTED_IDP
from saml2.samlp import STATUS_PARTIAL_LOGOUT
from saml2.samlp import STATUS_PROXY_COUNT_EXCEEDED
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_REQUEST_UNSUPPORTED
from saml2.samlp import STATUS_REQUEST_VERSION_DEPRECATED
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_HIGH
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_LOW
from saml2.samlp import STATUS_RESOURCE_NOT_RECOGNIZED
from saml2.samlp import STATUS_RESPONDER
from saml2.samlp import STATUS_TOO_MANY_RESPONSES
from saml2.samlp import STATUS_UNKNOWN_ATTR_PROFILE
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.samlp import STATUS_UNSUPPORTED_BINDING
from saml2.samlp import STATUS_VERSION_MISMATCH
from saml2.sigver import DecryptError
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import signed
from saml2.time_util import later_than
from saml2.time_util import str_to_time
from saml2.validate import NotValid
from saml2.validate import valid_address
from saml2.validate import valid_instance
from saml2.validate import validate_before
from saml2.validate import validate_on_or_after
class StatusResponse:
    msgtype = 'status_response'

    def __init__(self, sec_context, return_addrs=None, timeslack=0, request_id=0, asynchop=True, conv_info=None):
        self.sec = sec_context
        self.return_addrs = return_addrs or []
        self.timeslack = timeslack
        self.request_id = request_id
        self.xmlstr = ''
        self.origxml = ''
        self.name_id = None
        self.response = None
        self.not_on_or_after = 0
        self.in_response_to = None
        self.signature_check = self.sec.correctly_signed_response
        self.require_signature = False
        self.require_response_signature = False
        self.require_signature_or_response_signature = False
        self.not_signed = False
        self.asynchop = asynchop
        self.do_not_verify = False
        self.conv_info = conv_info or {}

    def _clear(self):
        self.xmlstr = ''
        self.name_id = None
        self.response = None
        self.not_on_or_after = 0

    def _postamble(self):
        if not self.response:
            logger.warning('Response was not correctly signed')
            if self.xmlstr:
                logger.debug('Response: %s', self.xmlstr)
            raise IncorrectlySigned()
        logger.debug('response: %s', self.response)
        try:
            valid_instance(self.response)
        except NotValid as exc:
            logger.warning('Not valid response: %s', exc.args[0])
            self._clear()
            return self
        self.in_response_to = self.response.in_response_to
        return self

    def load_instance(self, instance):
        if signed(instance):
            try:
                self.response = self.sec.check_signature(instance)
            except SignatureError:
                self.response = self.sec.check_signature(instance, f'{samlp.NAMESPACE}:Response')
        else:
            self.not_signed = True
            self.response = instance
        return self._postamble()

    def _loads(self, xmldata, decode=True, origxml=None):
        if isinstance(xmldata, bytes):
            self.xmlstr = xmldata[:].decode('utf-8')
        else:
            self.xmlstr = xmldata[:]
        logger.debug('xmlstr: %s', self.xmlstr)
        if origxml:
            self.origxml = origxml
        else:
            self.origxml = self.xmlstr
        if self.do_not_verify:
            args = {'do_not_verify': True}
        else:
            args = {}
        try:
            self.response = self.signature_check(xmldata, origdoc=origxml, must=self.require_signature, require_response_signature=self.require_response_signature, **args)
        except TypeError:
            raise
        except SignatureError:
            raise
        except Exception as excp:
            logger.exception('EXCEPTION: %s', str(excp))
            raise
        return self._postamble()

    def status_ok(self):
        status = self.response.status
        logger.debug('status: %s', status)
        if not status or status.status_code.value == samlp.STATUS_SUCCESS:
            return True
        err_code = status.status_code.status_code.value if status.status_code.status_code else None
        err_msg = status.status_message.text if status.status_message else err_code or 'Unknown error'
        err_cls = STATUSCODE2EXCEPTION.get(err_code, StatusError)
        msg = f'Unsuccessful operation: {status}\n{err_msg} from {err_code}'
        logger.debug(msg)
        raise err_cls(msg)

    def issue_instant_ok(self):
        """Check that the response was issued at a reasonable time"""
        upper = time_util.shift_time(time_util.time_in_a_while(days=1), self.timeslack).timetuple()
        lower = time_util.shift_time(time_util.time_a_while_ago(days=1), -self.timeslack).timetuple()
        issued_at = str_to_time(self.response.issue_instant)
        return lower < issued_at < upper

    def _verify(self):
        if self.request_id and self.in_response_to and (self.in_response_to != self.request_id):
            logger.error('Not the id I expected: %s != %s', self.in_response_to, self.request_id)
            return None
        if self.response.version != '2.0':
            _ver = float(self.response.version)
            if _ver < 2.0:
                raise RequestVersionTooLow()
            else:
                raise RequestVersionTooHigh()
        if self.asynchop:
            if self.response.destination and self.response.destination not in self.return_addrs:
                logger.error("destination '%s' not in return addresses '%s'", self.response.destination, self.return_addrs)
                return None
        valid = self.issue_instant_ok() and self.status_ok()
        return valid

    def loads(self, xmldata, decode=True, origxml=None):
        return self._loads(xmldata, decode, origxml)

    def verify(self, keys=None):
        try:
            return self._verify()
        except AssertionError:
            logger.exception('verify')
            return None

    def update(self, mold):
        self.xmlstr = mold.xmlstr
        self.in_response_to = mold.in_response_to
        self.response = mold.response

    def issuer(self):
        issuer_value = (self.response.issuer.text if self.response.issuer is not None else '').strip()
        return issuer_value