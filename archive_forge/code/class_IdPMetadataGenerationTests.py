import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class IdPMetadataGenerationTests(test_v3.RestfulTestCase):
    """A class for testing Identity Provider Metadata generation."""
    METADATA_URL = '/OS-FEDERATION/saml2/metadata'

    def setUp(self):
        super(IdPMetadataGenerationTests, self).setUp()
        self.generator = keystone_idp.MetadataGenerator()

    def config_overrides(self):
        super(IdPMetadataGenerationTests, self).config_overrides()
        self.config_fixture.config(group='saml', idp_entity_id=federation_fixtures.IDP_ENTITY_ID, idp_sso_endpoint=federation_fixtures.IDP_SSO_ENDPOINT, idp_organization_name=federation_fixtures.IDP_ORGANIZATION_NAME, idp_organization_display_name=federation_fixtures.IDP_ORGANIZATION_DISPLAY_NAME, idp_organization_url=federation_fixtures.IDP_ORGANIZATION_URL, idp_contact_company=federation_fixtures.IDP_CONTACT_COMPANY, idp_contact_name=federation_fixtures.IDP_CONTACT_GIVEN_NAME, idp_contact_surname=federation_fixtures.IDP_CONTACT_SURNAME, idp_contact_email=federation_fixtures.IDP_CONTACT_EMAIL, idp_contact_telephone=federation_fixtures.IDP_CONTACT_TELEPHONE_NUMBER, idp_contact_type=federation_fixtures.IDP_CONTACT_TYPE)

    def test_check_entity_id(self):
        metadata = self.generator.generate_metadata()
        self.assertEqual(federation_fixtures.IDP_ENTITY_ID, metadata.entity_id)

    def test_metadata_validity(self):
        """Call md.EntityDescriptor method that does internal verification."""
        self.generator.generate_metadata().verify()

    def test_serialize_metadata_object(self):
        """Check whether serialization doesn't raise any exceptions."""
        self.generator.generate_metadata().to_string()

    def test_check_idp_sso(self):
        metadata = self.generator.generate_metadata()
        idpsso_descriptor = metadata.idpsso_descriptor
        self.assertIsNotNone(metadata.idpsso_descriptor)
        self.assertEqual(federation_fixtures.IDP_SSO_ENDPOINT, idpsso_descriptor.single_sign_on_service.location)
        self.assertIsNotNone(idpsso_descriptor.organization)
        organization = idpsso_descriptor.organization
        self.assertEqual(federation_fixtures.IDP_ORGANIZATION_DISPLAY_NAME, organization.organization_display_name.text)
        self.assertEqual(federation_fixtures.IDP_ORGANIZATION_NAME, organization.organization_name.text)
        self.assertEqual(federation_fixtures.IDP_ORGANIZATION_URL, organization.organization_url.text)
        self.assertIsNotNone(idpsso_descriptor.contact_person)
        contact_person = idpsso_descriptor.contact_person
        self.assertEqual(federation_fixtures.IDP_CONTACT_GIVEN_NAME, contact_person.given_name.text)
        self.assertEqual(federation_fixtures.IDP_CONTACT_SURNAME, contact_person.sur_name.text)
        self.assertEqual(federation_fixtures.IDP_CONTACT_EMAIL, contact_person.email_address.text)
        self.assertEqual(federation_fixtures.IDP_CONTACT_TELEPHONE_NUMBER, contact_person.telephone_number.text)
        self.assertEqual(federation_fixtures.IDP_CONTACT_TYPE, contact_person.contact_type)

    def test_metadata_no_organization(self):
        self.config_fixture.config(group='saml', idp_organization_display_name=None, idp_organization_url=None, idp_organization_name=None)
        metadata = self.generator.generate_metadata()
        idpsso_descriptor = metadata.idpsso_descriptor
        self.assertIsNotNone(metadata.idpsso_descriptor)
        self.assertIsNone(idpsso_descriptor.organization)
        self.assertIsNotNone(idpsso_descriptor.contact_person)

    def test_metadata_no_contact_person(self):
        self.config_fixture.config(group='saml', idp_contact_name=None, idp_contact_surname=None, idp_contact_email=None, idp_contact_telephone=None)
        metadata = self.generator.generate_metadata()
        idpsso_descriptor = metadata.idpsso_descriptor
        self.assertIsNotNone(metadata.idpsso_descriptor)
        self.assertIsNotNone(idpsso_descriptor.organization)
        self.assertEqual([], idpsso_descriptor.contact_person)

    def test_metadata_invalid_idp_sso_endpoint(self):
        self.config_fixture.config(group='saml', idp_sso_endpoint=None)
        self.assertRaises(exception.ValidationError, self.generator.generate_metadata)

    def test_metadata_invalid_idp_entity_id(self):
        self.config_fixture.config(group='saml', idp_entity_id=None)
        self.assertRaises(exception.ValidationError, self.generator.generate_metadata)

    def test_get_metadata_with_no_metadata_file_configured(self):
        self.get(self.METADATA_URL, expected_status=http.client.INTERNAL_SERVER_ERROR)

    def test_get_head_metadata(self):
        self.config_fixture.config(group='saml', idp_metadata_path=XMLDIR + '/idp_saml2_metadata.xml')
        self.head(self.METADATA_URL, expected_status=http.client.OK)
        r = self.get(self.METADATA_URL, response_content_type='text/xml')
        self.assertEqual('text/xml', r.headers.get('Content-Type'))
        reference_file = _load_xml('idp_saml2_metadata.xml')
        reference_file = str.encode(reference_file)
        self.assertEqual(reference_file, r.result)