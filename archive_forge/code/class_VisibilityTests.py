import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class VisibilityTests(object):

    def test_unknown_admin_sees_all_but_community(self):
        images = self.db_api.image_get_all(self.admin_none_context)
        self.assertEqual(12, len(images))

    def test_unknown_admin_is_public_true(self):
        images = self.db_api.image_get_all(self.admin_none_context, is_public=True)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_admin_is_public_false(self):
        images = self.db_api.image_get_all(self.admin_none_context, is_public=False)
        self.assertEqual(8, len(images))
        for i in images:
            self.assertIn(i['visibility'], ['shared', 'private'])

    def test_unknown_admin_is_public_none(self):
        images = self.db_api.image_get_all(self.admin_none_context)
        self.assertEqual(12, len(images))

    def test_unknown_admin_visibility_public(self):
        images = self.db_api.image_get_all(self.admin_none_context, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_admin_visibility_shared(self):
        images = self.db_api.image_get_all(self.admin_none_context, filters={'visibility': 'shared'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('shared', i['visibility'])

    def test_unknown_admin_visibility_private(self):
        images = self.db_api.image_get_all(self.admin_none_context, filters={'visibility': 'private'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('private', i['visibility'])

    def test_unknown_admin_visibility_community(self):
        images = self.db_api.image_get_all(self.admin_none_context, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('community', i['visibility'])

    def test_unknown_admin_visibility_all(self):
        images = self.db_api.image_get_all(self.admin_none_context, filters={'visibility': 'all'})
        self.assertEqual(16, len(images))

    def test_known_admin_sees_all_but_others_community_images(self):
        images = self.db_api.image_get_all(self.admin_context)
        self.assertEqual(13, len(images))

    def test_known_admin_is_public_true(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=True)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_known_admin_is_public_false(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=False)
        self.assertEqual(9, len(images))
        for i in images:
            self.assertIn(i['visibility'], ['shared', 'private', 'community'])

    def test_known_admin_is_public_none(self):
        images = self.db_api.image_get_all(self.admin_context)
        self.assertEqual(13, len(images))

    def test_admin_as_user_true(self):
        images = self.db_api.image_get_all(self.admin_context, admin_as_user=True)
        self.assertEqual(7, len(images))
        for i in images:
            self.assertTrue('public' == i['visibility'] or i['owner'] == self.admin_tenant)

    def test_known_admin_visibility_public(self):
        images = self.db_api.image_get_all(self.admin_context, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_known_admin_visibility_shared(self):
        images = self.db_api.image_get_all(self.admin_context, filters={'visibility': 'shared'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('shared', i['visibility'])

    def test_known_admin_visibility_private(self):
        images = self.db_api.image_get_all(self.admin_context, filters={'visibility': 'private'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('private', i['visibility'])

    def test_known_admin_visibility_community(self):
        images = self.db_api.image_get_all(self.admin_context, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('community', i['visibility'])

    def test_known_admin_visibility_all(self):
        images = self.db_api.image_get_all(self.admin_context, filters={'visibility': 'all'})
        self.assertEqual(16, len(images))

    def test_what_unknown_user_sees(self):
        images = self.db_api.image_get_all(self.none_context)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_user_is_public_true(self):
        images = self.db_api.image_get_all(self.none_context, is_public=True)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_user_is_public_false(self):
        images = self.db_api.image_get_all(self.none_context, is_public=False)
        self.assertEqual(0, len(images))

    def test_unknown_user_is_public_none(self):
        images = self.db_api.image_get_all(self.none_context)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_user_visibility_public(self):
        images = self.db_api.image_get_all(self.none_context, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_unknown_user_visibility_shared(self):
        images = self.db_api.image_get_all(self.none_context, filters={'visibility': 'shared'})
        self.assertEqual(0, len(images))

    def test_unknown_user_visibility_private(self):
        images = self.db_api.image_get_all(self.none_context, filters={'visibility': 'private'})
        self.assertEqual(0, len(images))

    def test_unknown_user_visibility_community(self):
        images = self.db_api.image_get_all(self.none_context, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('community', i['visibility'])

    def test_unknown_user_visibility_all(self):
        images = self.db_api.image_get_all(self.none_context, filters={'visibility': 'all'})
        self.assertEqual(8, len(images))

    def test_what_tenant1_sees(self):
        images = self.db_api.image_get_all(self.tenant1_context)
        self.assertEqual(7, len(images))
        for i in images:
            if not 'public' == i['visibility']:
                self.assertEqual(i['owner'], self.tenant1)

    def test_tenant1_is_public_true(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=True)
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_tenant1_is_public_false(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=False)
        self.assertEqual(3, len(images))
        for i in images:
            self.assertEqual(i['owner'], self.tenant1)
            self.assertIn(i['visibility'], ['private', 'shared', 'community'])

    def test_tenant1_is_public_none(self):
        images = self.db_api.image_get_all(self.tenant1_context)
        self.assertEqual(7, len(images))
        for i in images:
            if not 'public' == i['visibility']:
                self.assertEqual(self.tenant1, i['owner'])

    def test_tenant1_visibility_public(self):
        images = self.db_api.image_get_all(self.tenant1_context, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('public', i['visibility'])

    def test_tenant1_visibility_shared(self):
        images = self.db_api.image_get_all(self.tenant1_context, filters={'visibility': 'shared'})
        self.assertEqual(1, len(images))
        self.assertEqual('shared', images[0]['visibility'])
        self.assertEqual(self.tenant1, images[0]['owner'])

    def test_tenant1_visibility_private(self):
        images = self.db_api.image_get_all(self.tenant1_context, filters={'visibility': 'private'})
        self.assertEqual(1, len(images))
        self.assertEqual('private', images[0]['visibility'])
        self.assertEqual(self.tenant1, images[0]['owner'])

    def test_tenant1_visibility_community(self):
        images = self.db_api.image_get_all(self.tenant1_context, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))
        for i in images:
            self.assertEqual('community', i['visibility'])

    def test_tenant1_visibility_all(self):
        images = self.db_api.image_get_all(self.tenant1_context, filters={'visibility': 'all'})
        self.assertEqual(10, len(images))

    def _setup_is_public_red_herring(self):
        values = {'name': 'Red Herring', 'owner': self.tenant1, 'visibility': 'shared', 'properties': {'is_public': 'silly'}}
        fixture = build_image_fixture(**values)
        self.db_api.image_create(self.admin_context, fixture)

    def test_is_public_is_a_normal_filter_for_admin(self):
        self._setup_is_public_red_herring()
        images = self.db_api.image_get_all(self.admin_context, filters={'is_public': 'silly'})
        self.assertEqual(1, len(images))
        self.assertEqual('Red Herring', images[0]['name'])

    def test_is_public_is_a_normal_filter_for_user(self):
        self._setup_is_public_red_herring()
        images = self.db_api.image_get_all(self.tenant1_context, filters={'is_public': 'silly'})
        self.assertEqual(1, len(images))
        self.assertEqual('Red Herring', images[0]['name'])

    def test_admin_is_public_true_and_visibility_public(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=True, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))

    def test_admin_is_public_false_and_visibility_public(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=False, filters={'visibility': 'public'})
        self.assertEqual(0, len(images))

    def test_admin_is_public_true_and_visibility_shared(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=True, filters={'visibility': 'shared'})
        self.assertEqual(0, len(images))

    def test_admin_is_public_false_and_visibility_shared(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=False, filters={'visibility': 'shared'})
        self.assertEqual(4, len(images))

    def test_admin_is_public_true_and_visibility_private(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=True, filters={'visibility': 'private'})
        self.assertEqual(0, len(images))

    def test_admin_is_public_false_and_visibility_private(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=False, filters={'visibility': 'private'})
        self.assertEqual(4, len(images))

    def test_admin_is_public_true_and_visibility_community(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=True, filters={'visibility': 'community'})
        self.assertEqual(0, len(images))

    def test_admin_is_public_false_and_visibility_community(self):
        images = self.db_api.image_get_all(self.admin_context, is_public=False, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))

    def test_tenant1_is_public_true_and_visibility_public(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=True, filters={'visibility': 'public'})
        self.assertEqual(4, len(images))

    def test_tenant1_is_public_false_and_visibility_public(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=False, filters={'visibility': 'public'})
        self.assertEqual(0, len(images))

    def test_tenant1_is_public_true_and_visibility_shared(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=True, filters={'visibility': 'shared'})
        self.assertEqual(0, len(images))

    def test_tenant1_is_public_false_and_visibility_shared(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=False, filters={'visibility': 'shared'})
        self.assertEqual(1, len(images))

    def test_tenant1_is_public_true_and_visibility_private(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=True, filters={'visibility': 'private'})
        self.assertEqual(0, len(images))

    def test_tenant1_is_public_false_and_visibility_private(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=False, filters={'visibility': 'private'})
        self.assertEqual(1, len(images))

    def test_tenant1_is_public_true_and_visibility_community(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=True, filters={'visibility': 'community'})
        self.assertEqual(0, len(images))

    def test_tenant1_is_public_false_and_visibility_community(self):
        images = self.db_api.image_get_all(self.tenant1_context, is_public=False, filters={'visibility': 'community'})
        self.assertEqual(4, len(images))