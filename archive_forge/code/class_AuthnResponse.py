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
class AuthnResponse(StatusResponse):
    """This is where all the profile compliance is checked.
    This one does saml2int compliance."""
    msgtype = 'authn_response'

    def __init__(self, sec_context, attribute_converters, entity_id, return_addrs=None, outstanding_queries=None, timeslack=0, asynchop=True, allow_unsolicited=False, test=False, allow_unknown_attributes=False, want_assertions_signed=False, want_assertions_or_response_signed=False, want_response_signed=False, conv_info=None, **kwargs):
        StatusResponse.__init__(self, sec_context, return_addrs, timeslack, asynchop=asynchop, conv_info=conv_info)
        self.entity_id = entity_id
        self.attribute_converters = attribute_converters
        if outstanding_queries:
            self.outstanding_queries = outstanding_queries
        else:
            self.outstanding_queries = {}
        self.context = 'AuthnReq'
        self.came_from = None
        self.ava = None
        self.assertion = None
        self.assertions = []
        self.session_not_on_or_after = 0
        self.allow_unsolicited = allow_unsolicited
        self.require_signature = want_assertions_signed
        self.require_signature_or_response_signature = want_assertions_or_response_signed
        self.require_response_signature = want_response_signed
        self.test = test
        self.allow_unknown_attributes = allow_unknown_attributes
        try:
            self.extension_schema = kwargs['extension_schema']
        except KeyError:
            self.extension_schema = {}

    def check_subject_confirmation_in_response_to(self, irp):
        for assertion in self.response.assertion:
            for _sc in assertion.subject.subject_confirmation:
                if _sc.subject_confirmation_data.in_response_to != irp:
                    return False
        return True

    def loads(self, xmldata, decode=True, origxml=None):
        self._loads(xmldata, decode, origxml)
        if self.asynchop:
            if self.in_response_to in self.outstanding_queries:
                self.came_from = self.outstanding_queries[self.in_response_to]
                try:
                    if not self.check_subject_confirmation_in_response_to(self.in_response_to):
                        raise UnsolicitedResponse(f'Unsolicited response: {self.in_response_to}')
                except AttributeError:
                    pass
            elif self.allow_unsolicited:
                pass
            else:
                raise UnsolicitedResponse(f'Unsolicited response: {self.in_response_to}')
        return self

    def clear(self):
        self._clear()
        self.came_from = None
        self.ava = None
        self.assertion = None

    def authn_statement_ok(self, optional=False):
        n_authn_statements = len(self.assertion.authn_statement)
        if n_authn_statements != 1:
            if optional:
                return True
            else:
                msg = f'Invalid number of AuthnStatement found in Response: {n_authn_statements}'
                raise ValueError(msg)
        authn_statement = self.assertion.authn_statement[0]
        if authn_statement.session_not_on_or_after:
            if validate_on_or_after(authn_statement.session_not_on_or_after, self.timeslack):
                self.session_not_on_or_after = calendar.timegm(time_util.str_to_time(authn_statement.session_not_on_or_after))
            else:
                return False
        return True

    def condition_ok(self, lax=False):
        if not self.assertion.conditions:
            return True
        if self.test:
            lax = True
        conditions = self.assertion.conditions
        logger.debug('conditions: %s', conditions)
        if not conditions.keyswv():
            return True
        if conditions.not_before and conditions.not_on_or_after:
            if not later_than(conditions.not_on_or_after, conditions.not_before):
                return False
        try:
            if conditions.not_on_or_after:
                self.not_on_or_after = validate_on_or_after(conditions.not_on_or_after, self.timeslack)
            if conditions.not_before:
                validate_before(conditions.not_before, self.timeslack)
        except Exception as excp:
            logger.error('Exception on conditions: %s', str(excp))
            if not lax:
                raise
            else:
                self.not_on_or_after = 0
        if not for_me(conditions, self.entity_id):
            if not lax:
                raise Exception(f'AudienceRestrictions conditions not satisfied! (Local entity_id={self.entity_id})')
        if conditions.condition:
            for cond in conditions.condition:
                try:
                    if cond.extension_attributes[XSI_TYPE] in self.extension_schema:
                        pass
                    else:
                        raise Exception('Unknown condition')
                except KeyError:
                    raise Exception('Missing xsi:type specification')
        return True

    def decrypt_attributes(self, attribute_statement, keys=None):
        """
        Decrypts possible encrypted attributes and adds the decrypts to the
        list of attributes.

        :param attribute_statement: A SAML.AttributeStatement which might
            contain both encrypted attributes and attributes.
        """
        for encattr in attribute_statement.encrypted_attribute:
            if not encattr.encrypted_key:
                _decr = self.sec.decrypt_keys(encattr.encrypted_data, keys=keys)
                _attr = attribute_from_string(_decr)
                attribute_statement.attribute.append(_attr)
            else:
                _decr = self.sec.decrypt_keys(encattr, keys=keys)
                enc_attr = encrypted_attribute_from_string(_decr)
                attrlist = enc_attr.extensions_as_elements('Attribute', saml)
                attribute_statement.attribute.extend(attrlist)

    def read_attribute_statement(self, attr_statem):
        logger.debug('Attribute Statement: %s', attr_statem)
        self.decrypt_attributes(attr_statem)
        return to_local(self.attribute_converters, attr_statem, self.allow_unknown_attributes)

    def get_identity(self):
        """The assertion can contain zero or more attributeStatements"""
        ava = {}
        for _assertion in self.assertions:
            if _assertion.advice:
                if _assertion.advice.assertion:
                    for tmp_assertion in _assertion.advice.assertion:
                        if tmp_assertion.attribute_statement:
                            n_attr_statements = len(tmp_assertion.attribute_statement)
                            if n_attr_statements != 1:
                                msg = 'Invalid number of AuthnStatement found in Response: {n}'.format(n=n_attr_statements)
                                raise ValueError(msg)
                            ava.update(self.read_attribute_statement(tmp_assertion.attribute_statement[0]))
            if _assertion.attribute_statement:
                logger.debug('Assertion contains %s attribute statement(s)', len(self.assertion.attribute_statement))
                for _attr_statem in _assertion.attribute_statement:
                    logger.debug(f'Attribute Statement: {_attr_statem}')
                    ava.update(self.read_attribute_statement(_attr_statem))
            if not ava:
                logger.debug('Assertion contains no attribute statements')
        return ava

    def _bearer_confirmed(self, data):
        if not data:
            return False
        if data.address:
            if not valid_address(data.address):
                return False
        validate_on_or_after(data.not_on_or_after, self.timeslack)
        validate_before(data.not_before, self.timeslack)
        if not later_than(data.not_on_or_after, data.not_before):
            return False
        if self.asynchop and self.came_from is None:
            if data.in_response_to:
                if data.in_response_to in self.outstanding_queries:
                    self.came_from = self.outstanding_queries[data.in_response_to]
                elif self.allow_unsolicited:
                    pass
                else:
                    logger.debug("in response to: '%s'", data.in_response_to)
                    logger.info('outstanding queries: %s', self.outstanding_queries.keys())
                    raise Exception("Combination of session id and requestURI I don't recall")
        return True

    def _holder_of_key_confirmed(self, data):
        if not data or not data.extension_elements:
            return False
        has_keyinfo = False
        for element in extension_elements_to_elements(data.extension_elements, [samlp, saml, xenc, ds]):
            if isinstance(element, ds.KeyInfo):
                has_keyinfo = True
        return has_keyinfo

    def get_subject(self, keys=None):
        """The assertion must contain a Subject"""
        if not self.assertion:
            raise ValueError('Missing assertion')
        if not self.assertion.subject:
            raise ValueError(f'Invalid assertion subject: {self.assertion.subject}')
        subject = self.assertion.subject
        subjconf = []
        if not self.verify_attesting_entity(subject.subject_confirmation):
            raise VerificationError('No valid attesting address')
        for subject_confirmation in subject.subject_confirmation:
            _data = subject_confirmation.subject_confirmation_data
            if subject_confirmation.method == SCM_BEARER:
                if not self._bearer_confirmed(_data):
                    continue
            elif subject_confirmation.method == SCM_HOLDER_OF_KEY:
                if not self._holder_of_key_confirmed(_data):
                    continue
            elif subject_confirmation.method == SCM_SENDER_VOUCHES:
                pass
            else:
                raise ValueError(f'Unknown subject confirmation method: {subject_confirmation.method}')
            _recip = _data.recipient
            if not _recip or not self.verify_recipient(_recip):
                raise VerificationError('No valid recipient')
            subjconf.append(subject_confirmation)
        if not subjconf:
            raise VerificationError('No valid subject confirmation')
        subject.subject_confirmation = subjconf
        if subject.name_id:
            self.name_id = subject.name_id
        elif subject.encrypted_id:
            _name_id_str = self.sec.decrypt_keys(subject.encrypted_id.encrypted_data.to_string(), keys=keys)
            _name_id = saml.name_id_from_string(_name_id_str)
            self.name_id = _name_id
        logger.info('Subject NameID: %s', self.name_id)
        return self.name_id

    def _assertion(self, assertion, verified=False):
        """
        Check the assertion
        :param assertion:
        :return: True/False depending on if the assertion is sane or not
        """
        if not hasattr(assertion, 'signature') or not assertion.signature:
            logger.debug('unsigned')
            if self.require_signature:
                raise SignatureError('Signature missing for assertion')
        else:
            logger.debug('signed')
            if not verified and self.do_not_verify is False:
                try:
                    self.sec.check_signature(assertion, class_name(assertion), self.xmlstr)
                except Exception as exc:
                    logger.error('The signature on the assertion cannot be verified.')
                    logger.debug('correctly_signed_response: %s', exc)
                    raise
        self.assertion = assertion
        logger.debug('assertion context: %s', self.context)
        logger.debug('assertion keys: %s', assertion.keyswv())
        logger.debug('outstanding_queries: %s', self.outstanding_queries)
        if self.context == 'AuthnReq':
            self.authn_statement_ok()
        if not self.condition_ok():
            raise VerificationError('Condition not OK')
        logger.debug('--- Getting Identity ---')
        try:
            self.get_subject()
            if self.asynchop:
                if self.allow_unsolicited:
                    pass
                elif self.came_from is None:
                    raise VerificationError('Came from')
            return True
        except Exception:
            logger.exception('get subject')
            raise

    def decrypt_assertions(self, encrypted_assertions, decr_txt, issuer=None, verified=False):
        """Moves the decrypted assertion from the encrypted assertion to a
        list.

        :param encrypted_assertions: A list of encrypted assertions.
        :param decr_txt: The string representation containing the decrypted
        data. Used when verifying signatures.
        :param issuer: The issuer of the response.
        :param verified: If True do not verify signatures, otherwise verify
        the signature if it exists.
        :return: A list of decrypted assertions.
        """
        res = []
        for encrypted_assertion in encrypted_assertions:
            if encrypted_assertion.extension_elements:
                assertions = extension_elements_to_elements(encrypted_assertion.extension_elements, [saml, samlp])
                for assertion in assertions:
                    if assertion.signature and (not verified):
                        if not self.sec.check_signature(assertion, origdoc=decr_txt, node_name=class_name(assertion), issuer=issuer):
                            logger.error("Failed to verify signature on '%s'", assertion)
                            raise SignatureError()
                    res.append(assertion)
        return res

    def find_encrypt_data_assertion(self, enc_assertions):
        """Verifies if a list of encrypted assertions contains encrypted data.

        :param enc_assertions: A list of encrypted assertions.
        :return: True encrypted data exists otherwise false.
        """
        for _assertion in enc_assertions:
            if _assertion.encrypted_data is not None:
                return True

    def find_encrypt_data_assertion_list(self, _assertions):
        """Verifies if a list of assertions contains encrypted data in the
        advice element.

        :param _assertions: A list of assertions.
        :return: True encrypted data exists otherwise false.
        """
        for _assertion in _assertions:
            if _assertion.advice:
                if _assertion.advice.encrypted_assertion:
                    res = self.find_encrypt_data_assertion(_assertion.advice.encrypted_assertion)
                    if res:
                        return True

    def find_encrypt_data(self, resp):
        """Verifies if a saml response contains encrypted assertions with
        encrypted data.

        :param resp: A saml response.
        :return: True encrypted data exists otherwise false.
        """
        if resp.encrypted_assertion:
            res = self.find_encrypt_data_assertion(resp.encrypted_assertion)
            if res:
                return True
        if resp.assertion:
            for tmp_assertion in resp.assertion:
                if tmp_assertion.advice:
                    if tmp_assertion.advice.encrypted_assertion:
                        res = self.find_encrypt_data_assertion(tmp_assertion.advice.encrypted_assertion)
                        if res:
                            return True
        return False

    def parse_assertion(self, keys=None):
        """Parse the assertions for a saml response.

        :param keys: A string representing a RSA key or a list of strings
        containing RSA keys.
        :return: True if the assertions are parsed otherwise False.
        """
        if self.context == 'AuthnQuery':
            pass
        else:
            n_assertions = len(self.response.assertion)
            n_assertions_enc = len(self.response.encrypted_assertion)
            if n_assertions != 1 and n_assertions_enc != 1 and (self.assertion is None):
                raise InvalidAssertion(f'Invalid number of assertions in Response: {n_assertions + n_assertions_enc}')
        if self.response.assertion:
            logger.debug('***Unencrypted assertion***')
            for assertion in self.response.assertion:
                if not self._assertion(assertion, False):
                    return False
        if self.find_encrypt_data(self.response):
            logger.debug('***Encrypted assertion/-s***')
            _enc_assertions = []
            resp = self.response
            decr_text = str(self.response)
            decr_text_old = None
            while self.find_encrypt_data(resp) and decr_text_old != decr_text:
                decr_text_old = decr_text
                try:
                    decr_text = self.sec.decrypt_keys(decr_text, keys=keys)
                except DecryptError:
                    continue
                else:
                    resp = samlp.response_from_string(decr_text)
                    if type(decr_text_old) != type(decr_text):
                        if isinstance(decr_text_old, bytes):
                            decr_text_old = decr_text_old.decode('utf-8')
                        else:
                            decr_text_old = decr_text_old.encode('utf-8')
            _enc_assertions = self.decrypt_assertions(resp.encrypted_assertion, decr_text)
            decr_text_old = None
            while (self.find_encrypt_data(resp) or self.find_encrypt_data_assertion_list(_enc_assertions)) and decr_text_old != decr_text:
                decr_text_old = decr_text
                try:
                    decr_text = self.sec.decrypt_keys(decr_text, keys=keys)
                except DecryptError:
                    continue
                else:
                    resp = samlp.response_from_string(decr_text)
                    _enc_assertions = self.decrypt_assertions(resp.encrypted_assertion, decr_text, verified=True)
                    if type(decr_text_old) != type(decr_text):
                        if isinstance(decr_text_old, bytes):
                            decr_text_old = decr_text_old.decode('utf-8')
                        else:
                            decr_text_old = decr_text_old.encode('utf-8')
            all_assertions = _enc_assertions
            if resp.assertion:
                all_assertions = all_assertions + resp.assertion
            if len(all_assertions) > 0:
                for tmp_ass in all_assertions:
                    if tmp_ass.advice and tmp_ass.advice.encrypted_assertion:
                        advice_res = self.decrypt_assertions(tmp_ass.advice.encrypted_assertion, decr_text, tmp_ass.issuer)
                        if tmp_ass.advice.assertion:
                            tmp_ass.advice.assertion.extend(advice_res)
                        else:
                            tmp_ass.advice.assertion = advice_res
                        if len(advice_res) > 0:
                            tmp_ass.advice.encrypted_assertion = []
            self.response.assertion = resp.assertion
            for assertion in _enc_assertions:
                if not self._assertion(assertion, True):
                    return False
                else:
                    self.assertions.append(assertion)
            self.xmlstr = decr_text
            if len(_enc_assertions) > 0:
                self.response.encrypted_assertion = []
        if self.response.assertion:
            for assertion in self.response.assertion:
                self.assertions.append(assertion)
        if self.assertions and len(self.assertions) > 0:
            self.assertion = self.assertions[0]
        if self.context == 'AuthnReq' or self.context == 'AttrQuery':
            self.ava = self.get_identity()
            logger.debug(f'--- AVA: {self.ava}')
        return True

    def verify(self, keys=None):
        """Verify that the assertion is syntactically correct and the
        signature is correct if present.

        :param keys: If not the default key file should be used then use one
        of these.
        """
        try:
            res = self._verify()
        except AssertionError as err:
            logger.error('Verification error on the response: %s', str(err))
            raise
        else:
            if not res:
                return None
        if not isinstance(self.response, samlp.Response):
            return self
        if self.parse_assertion(keys):
            return self
        else:
            logger.error('Could not parse the assertion')
            return None

    def session_id(self):
        """Returns the SessionID of the response"""
        return self.response.in_response_to

    def id(self):
        """Return the ID of the response"""
        return self.response.id

    def authn_info(self):
        res = []
        for statement in getattr(self.assertion, 'authn_statement', []):
            authn_instant = getattr(statement, 'authn_instant', '')
            context = statement.authn_context
            if not context:
                continue
            authn_class = getattr(context.authn_context_class_ref, 'text', None) or getattr(context.authn_context_decl_ref, 'text', None) or ''
            authenticating_authorities = getattr(context, 'authenticating_authority', [])
            authn_auth = [authority.text for authority in authenticating_authorities]
            res.append((authn_class, authn_auth, authn_instant))
        return res

    def authz_decision_info(self):
        res = {'permit': [], 'deny': [], 'indeterminate': []}
        for adstat in self.assertion.authz_decision_statement:
            res[adstat.decision.text.lower()] = adstat
        return res

    def session_info(self):
        """Returns a predefined set of information gleened from the
        response.
        :returns: Dictionary with information
        """
        if self.session_not_on_or_after > 0:
            nooa = self.session_not_on_or_after
        else:
            nooa = self.not_on_or_after
        if self.context == 'AuthzQuery':
            return {'name_id': self.name_id, 'came_from': self.came_from, 'issuer': self.issuer(), 'not_on_or_after': nooa, 'authz_decision_info': self.authz_decision_info()}
        elif getattr(self.assertion, 'authn_statement', None):
            authn_statement = self.assertion.authn_statement[0]
            return {'ava': self.ava, 'name_id': self.name_id, 'came_from': self.came_from, 'issuer': self.issuer(), 'not_on_or_after': nooa, 'authn_info': self.authn_info(), 'session_index': authn_statement.session_index}
        else:
            raise StatusInvalidAuthnResponseStatement('The Authn Response Statement is not valid')

    def __str__(self):
        return self.xmlstr

    def verify_recipient(self, recipient):
        """
        Verify that I'm the recipient of the assertion

        :param recipient: A URI specifying the entity or location to which an
            attesting entity can present the assertion.
        :return: True/False
        """
        if not self.conv_info:
            return True
        _info = self.conv_info
        try:
            if recipient == _info['entity_id']:
                return True
        except KeyError:
            pass
        try:
            if recipient in self.return_addrs:
                return True
        except KeyError:
            pass
        return False

    def verify_attesting_entity(self, subject_confirmation):
        """
        At least one address specification has to be correct.

        :param subject_confirmation: A SubbjectConfirmation instance
        :return: True/False
        """
        try:
            address = self.conv_info['remote_addr']
        except KeyError:
            address = '0.0.0.0'
        correct = 0
        for subject_conf in subject_confirmation:
            if subject_conf.subject_confirmation_data is None:
                correct += 1
            elif subject_conf.subject_confirmation_data.address:
                if address == '0.0.0.0':
                    correct += 1
                elif subject_conf.subject_confirmation_data.address == address:
                    correct += 1
            else:
                correct += 1
        if correct:
            return True
        else:
            return False