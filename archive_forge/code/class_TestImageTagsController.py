import http.client as http
import webob
import glance.api.v2.image_tags
from glance.common import exception
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.unit.v2.test_image_data_resource as image_data_tests
import glance.tests.utils as test_utils
class TestImageTagsController(base.IsolatedUnitTest):

    def setUp(self):
        super(TestImageTagsController, self).setUp()
        self.db = unit_test_utils.FakeDB()
        self.controller = glance.api.v2.image_tags.Controller(self.db)

    def test_create_tag(self):
        request = unit_test_utils.get_fake_request()
        self.controller.update(request, unit_test_utils.UUID1, 'dink')
        context = request.context
        tags = self.db.image_tag_get_all(context, unit_test_utils.UUID1)
        self.assertEqual(1, len([tag for tag in tags if tag == 'dink']))

    def test_create_too_many_tags(self):
        self.config(image_tag_quota=0)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.update, request, unit_test_utils.UUID1, 'dink')

    def test_create_duplicate_tag_ignored(self):
        request = unit_test_utils.get_fake_request()
        self.controller.update(request, unit_test_utils.UUID1, 'dink')
        self.controller.update(request, unit_test_utils.UUID1, 'dink')
        context = request.context
        tags = self.db.image_tag_get_all(context, unit_test_utils.UUID1)
        self.assertEqual(1, len([tag for tag in tags if tag == 'dink']))

    def test_update_tag_of_non_existing_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.update, request, 'abcd', 'dink')

    def test_delete_tag_forbidden(self):

        def fake_get(self):
            raise exception.Forbidden()
        image_repo = image_data_tests.FakeImageRepo()
        image_repo.get = fake_get

        def get_fake_repo(self):
            return image_repo
        self.controller.gateway.get_repo = get_fake_repo
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, unit_test_utils.UUID1, 'ping')

    def test_delete_tag(self):
        request = unit_test_utils.get_fake_request()
        self.controller.delete(request, unit_test_utils.UUID1, 'ping')

    def test_delete_tag_not_found(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, unit_test_utils.UUID1, 'what')

    def test_delete_tag_of_non_existing_image(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, 'abcd', 'dink')