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
class MembershipVisibilityTests(object):

    def _check_by_member(self, ctx, member_id, expected):
        members = self.db_api.image_member_find(ctx, member=member_id)
        images = [self.db_api.image_get(self.admin_ctx, member['image_id']) for member in members]
        facets = [(image['owner'], image['name']) for image in images]
        self.assertEqual(set(expected), set(facets))

    def test_owner1_finding_user1_memberships(self):
        """Owner1 should see images it owns that are shared with User1."""
        expected = [(self.owner1, 'shared-with-1'), (self.owner1, 'shared-with-both')]
        self._check_by_member(self.owner1_ctx, self.tenant1, expected)

    def test_user1_finding_user1_memberships(self):
        """User1 should see all images shared with User1 """
        expected = [(self.owner1, 'shared-with-1'), (self.owner1, 'shared-with-both'), (self.owner2, 'shared-with-1'), (self.owner2, 'shared-with-both')]
        self._check_by_member(self.user1_ctx, self.tenant1, expected)

    def test_user2_finding_user1_memberships(self):
        """User2 should see no images shared with User1 """
        expected = []
        self._check_by_member(self.user2_ctx, self.tenant1, expected)

    def test_admin_finding_user1_memberships(self):
        """Admin should see all images shared with User1 """
        expected = [(self.owner1, 'shared-with-1'), (self.owner1, 'shared-with-both'), (self.owner2, 'shared-with-1'), (self.owner2, 'shared-with-both')]
        self._check_by_member(self.admin_ctx, self.tenant1, expected)

    def _check_by_image(self, context, image_id, expected):
        members = self.db_api.image_member_find(context, image_id=image_id)
        member_ids = [member['member'] for member in members]
        self.assertEqual(set(expected), set(member_ids))

    def test_owner1_finding_owner1s_image_members(self):
        """Owner1 should see all memberships of its image """
        expected = [self.tenant1, self.tenant2]
        image_id = self.image_ids[self.owner1, 'shared-with-both']
        self._check_by_image(self.owner1_ctx, image_id, expected)

    def test_admin_finding_owner1s_image_members(self):
        """Admin should see all memberships of owner1's image """
        expected = [self.tenant1, self.tenant2]
        image_id = self.image_ids[self.owner1, 'shared-with-both']
        self._check_by_image(self.admin_ctx, image_id, expected)

    def test_user1_finding_owner1s_image_members(self):
        """User1 should see its own membership of owner1's image """
        expected = [self.tenant1]
        image_id = self.image_ids[self.owner1, 'shared-with-both']
        self._check_by_image(self.user1_ctx, image_id, expected)

    def test_user2_finding_owner1s_image_members(self):
        """User2 should see its own membership of owner1's image """
        expected = [self.tenant2]
        image_id = self.image_ids[self.owner1, 'shared-with-both']
        self._check_by_image(self.user2_ctx, image_id, expected)

    def test_user3_finding_owner1s_image_members(self):
        """User3 should see no memberships of owner1's image """
        expected = []
        image_id = self.image_ids[self.owner1, 'shared-with-both']
        self._check_by_image(self.user3_ctx, image_id, expected)