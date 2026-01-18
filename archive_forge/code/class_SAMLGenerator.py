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
class SAMLGenerator(object):
    """A class to generate SAML assertions."""

    def __init__(self):
        self.assertion_id = uuid.uuid4().hex

    def samlize_token(self, issuer, recipient, user, user_domain_name, roles, project, project_domain_name, groups, expires_in=None):
        """Convert Keystone attributes to a SAML assertion.

        :param issuer: URL of the issuing party
        :type issuer: string
        :param recipient: URL of the recipient
        :type recipient: string
        :param user: User name
        :type user: string
        :param user_domain_name: User Domain name
        :type user_domain_name: string
        :param roles: List of role names
        :type roles: list
        :param project: Project name
        :type project: string
        :param project_domain_name: Project Domain name
        :type project_domain_name: string
        :param groups: List of strings of user groups and domain name, where
                       strings are serialized dictionaries.
        :type groups: list
        :param expires_in: Sets how long the assertion is valid for, in seconds
        :type expires_in: int

        :returns: XML <Response> object

        """
        expiration_time = self._determine_expiration_time(expires_in)
        status = self._create_status()
        saml_issuer = self._create_issuer(issuer)
        subject = self._create_subject(user, expiration_time, recipient)
        attribute_statement = self._create_attribute_statement(user, user_domain_name, roles, project, project_domain_name, groups)
        authn_statement = self._create_authn_statement(issuer, expiration_time)
        signature = self._create_signature()
        assertion = self._create_assertion(saml_issuer, signature, subject, authn_statement, attribute_statement)
        assertion = _sign_assertion(assertion)
        response = self._create_response(saml_issuer, status, assertion, recipient)
        return response

    def _determine_expiration_time(self, expires_in):
        if expires_in is None:
            expires_in = CONF.saml.assertion_expiration_time
        now = timeutils.utcnow()
        future = now + datetime.timedelta(seconds=expires_in)
        return utils.isotime(future, subsecond=True)

    def _create_status(self):
        """Create an object that represents a SAML Status.

        <ns0:Status xmlns:ns0="urn:oasis:names:tc:SAML:2.0:protocol">
            <ns0:StatusCode
              Value="urn:oasis:names:tc:SAML:2.0:status:Success" />
        </ns0:Status>

        :returns: XML <Status> object

        """
        status = samlp.Status()
        status_code = samlp.StatusCode()
        status_code.value = samlp.STATUS_SUCCESS
        status_code.set_text('')
        status.status_code = status_code
        return status

    def _create_issuer(self, issuer_url):
        """Create an object that represents a SAML Issuer.

        <ns0:Issuer
          xmlns:ns0="urn:oasis:names:tc:SAML:2.0:assertion"
          Format="urn:oasis:names:tc:SAML:2.0:nameid-format:entity">
          https://acme.com/FIM/sps/openstack/saml20</ns0:Issuer>

        :returns: XML <Issuer> object

        """
        issuer = saml.Issuer()
        issuer.format = saml.NAMEID_FORMAT_ENTITY
        issuer.set_text(issuer_url)
        return issuer

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

    def _create_attribute_statement(self, user, user_domain_name, roles, project, project_domain_name, groups):
        """Create an object that represents a SAML AttributeStatement.

        <ns0:AttributeStatement>
            <ns0:Attribute Name="openstack_user">
                <ns0:AttributeValue
                  xsi:type="xs:string">test_user</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_user_domain">
                <ns0:AttributeValue
                  xsi:type="xs:string">Default</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_roles">
                <ns0:AttributeValue
                  xsi:type="xs:string">admin</ns0:AttributeValue>
                <ns0:AttributeValue
                  xsi:type="xs:string">member</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_project">
                <ns0:AttributeValue
                  xsi:type="xs:string">development</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_project_domain">
                <ns0:AttributeValue
                  xsi:type="xs:string">Default</ns0:AttributeValue>
            </ns0:Attribute>
            <ns0:Attribute Name="openstack_groups">
                <ns0:AttributeValue
                   xsi:type="xs:string">JSON:{"name":"group1","domain":{"name":"Default"}}
                </ns0:AttributeValue>
                <ns0:AttributeValue
                   xsi:type="xs:string">JSON:{"name":"group2","domain":{"name":"Default"}}
                </ns0:AttributeValue>
            </ns0:Attribute>

        </ns0:AttributeStatement>

        :returns: XML <AttributeStatement> object

        """

        def _build_attribute(attribute_name, attribute_values):
            attribute = saml.Attribute()
            attribute.name = attribute_name
            for value in attribute_values:
                attribute_value = saml.AttributeValue()
                attribute_value.set_text(value)
                attribute.attribute_value.append(attribute_value)
            return attribute
        user_attribute = _build_attribute('openstack_user', [user])
        roles_attribute = _build_attribute('openstack_roles', roles)
        project_attribute = _build_attribute('openstack_project', [project])
        project_domain_attribute = _build_attribute('openstack_project_domain', [project_domain_name])
        user_domain_attribute = _build_attribute('openstack_user_domain', [user_domain_name])
        attribute_statement = saml.AttributeStatement()
        attribute_statement.attribute.append(user_attribute)
        attribute_statement.attribute.append(roles_attribute)
        attribute_statement.attribute.append(project_attribute)
        attribute_statement.attribute.append(project_domain_attribute)
        attribute_statement.attribute.append(user_domain_attribute)
        if groups:
            groups_attribute = _build_attribute('openstack_groups', groups)
            attribute_statement.attribute.append(groups_attribute)
        return attribute_statement

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

    def _create_assertion(self, issuer, signature, subject, authn_statement, attribute_statement):
        """Create an object that represents a SAML Assertion.

        <ns0:Assertion
          ID="35daed258ba647ba8962e9baff4d6a46"
          IssueInstant="2014-06-11T15:45:58Z"
          Version="2.0">
            <ns0:Issuer> ... </ns0:Issuer>
            <ns1:Signature> ... </ns1:Signature>
            <ns0:Subject> ... </ns0:Subject>
            <ns0:AuthnStatement> ... </ns0:AuthnStatement>
            <ns0:AttributeStatement> ... </ns0:AttributeStatement>
        </ns0:Assertion>

        :returns: XML <Assertion> object

        """
        assertion = saml.Assertion()
        assertion.id = self.assertion_id
        assertion.issue_instant = utils.isotime()
        assertion.version = '2.0'
        assertion.issuer = issuer
        assertion.signature = signature
        assertion.subject = subject
        assertion.authn_statement = authn_statement
        assertion.attribute_statement = attribute_statement
        return assertion

    def _create_response(self, issuer, status, assertion, recipient):
        """Create an object that represents a SAML Response.

        <ns0:Response
          Destination="http://beta.com/Shibboleth.sso/SAML2/POST"
          ID="c5954543230e4e778bc5b92923a0512d"
          IssueInstant="2014-07-30T03:19:45Z"
          Version="2.0" />
            <ns0:Issuer> ... </ns0:Issuer>
            <ns0:Assertion> ... </ns0:Assertion>
            <ns0:Status> ... </ns0:Status>
        </ns0:Response>

        :returns: XML <Response> object

        """
        response = samlp.Response()
        response.id = uuid.uuid4().hex
        response.destination = recipient
        response.issue_instant = utils.isotime()
        response.version = '2.0'
        response.issuer = issuer
        response.status = status
        response.assertion = assertion
        return response

    def _create_signature(self):
        """Create an object that represents a SAML <Signature>.

        This must be filled with algorithms that the signing binary will apply
        in order to sign the whole message.
        Currently we enforce X509 signing.
        Example of the template::

        <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
          <SignedInfo>
            <CanonicalizationMethod
              Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>
            <SignatureMethod
              Algorithm="http://www.w3.org/2000/09/xmldsig#rsa-sha1"/>
            <Reference URI="#<Assertion ID>">
              <Transforms>
                <Transform
            Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>
               <Transform Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>
              </Transforms>
             <DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>
             <DigestValue />
            </Reference>
          </SignedInfo>
          <SignatureValue />
          <KeyInfo>
            <X509Data />
          </KeyInfo>
        </Signature>

        :returns: XML <Signature> object

        """
        canonicalization_method = xmldsig.CanonicalizationMethod()
        if hasattr(xmldsig, 'TRANSFORM_C14N'):
            canonicalization_method.algorithm = xmldsig.TRANSFORM_C14N
        else:
            canonicalization_method.algorithm = xmldsig.ALG_EXC_C14N
        signature_method = xmldsig.SignatureMethod(algorithm=xmldsig.SIG_RSA_SHA1)
        transforms = xmldsig.Transforms()
        envelope_transform = xmldsig.Transform(algorithm=xmldsig.TRANSFORM_ENVELOPED)
        if hasattr(xmldsig, 'TRANSFORM_C14N'):
            c14_transform = xmldsig.Transform(algorithm=xmldsig.TRANSFORM_C14N)
        else:
            c14_transform = xmldsig.Transform(algorithm=xmldsig.ALG_EXC_C14N)
        transforms.transform = [envelope_transform, c14_transform]
        digest_method = xmldsig.DigestMethod(algorithm=xmldsig.DIGEST_SHA1)
        digest_value = xmldsig.DigestValue()
        reference = xmldsig.Reference()
        reference.uri = '#' + self.assertion_id
        reference.digest_method = digest_method
        reference.digest_value = digest_value
        reference.transforms = transforms
        signed_info = xmldsig.SignedInfo()
        signed_info.canonicalization_method = canonicalization_method
        signed_info.signature_method = signature_method
        signed_info.reference = reference
        key_info = xmldsig.KeyInfo()
        key_info.x509_data = xmldsig.X509Data()
        signature = xmldsig.Signature()
        signature.signed_info = signed_info
        signature.signature_value = xmldsig.SignatureValue()
        signature.key_info = key_info
        return signature