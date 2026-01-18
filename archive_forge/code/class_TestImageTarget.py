from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
class TestImageTarget(base.IsolatedUnitTest):

    def test_image_target_ignores_locations(self):
        image = ImageStub()
        target = glance.api.policy.ImageTarget(image)
        self.assertNotIn('locations', list(target))

    def test_image_target_project_id_alias(self):
        image = ImageStub()
        target = glance.api.policy.ImageTarget(image)
        self.assertIn('project_id', target)
        self.assertEqual(image.owner, target['project_id'])
        self.assertEqual(image.owner, target['owner'])

    def test_image_target_transforms(self):
        fake_image = mock.MagicMock()
        fake_image.image_id = mock.sentinel.image_id
        fake_image.owner = mock.sentinel.owner
        fake_image.member = mock.sentinel.member
        target = glance.api.policy.ImageTarget(fake_image)
        self.assertEqual(mock.sentinel.image_id, target['id'])
        self.assertEqual(mock.sentinel.owner, target['project_id'])
        self.assertEqual(mock.sentinel.member, target['member_id'])
        self.assertEqual(mock.sentinel.image_id, target['image_id'])
        self.assertEqual(mock.sentinel.owner, target['owner'])
        self.assertEqual(mock.sentinel.member, target['member'])