import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestCooldownMixin(common.HeatTestCase):

    def setUp(self):
        super(TestCooldownMixin, self).setUp()
        t = template_format.parse(inline_templates.as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stack.store()
        self.group = self.stack['WebServerGroup']
        self.group.state_set('CREATE', 'COMPLETE')

    def test_cooldown_is_in_progress_toosoon(self):
        cooldown_end = timeutils.utcnow() + datetime.timedelta(seconds=60)
        previous_meta = {'cooldown_end': {cooldown_end.isoformat(): 'ChangeInCapacity : 1'}}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertRaises(resource.NoActionRequired, self.group._check_scaling_allowed, 60)

    def test_cooldown_is_in_progress_toosoon_legacy(self):
        now = timeutils.utcnow()
        previous_meta = {'cooldown': {now.isoformat(): 'ChangeInCapacity : 1'}}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertRaises(resource.NoActionRequired, self.group._check_scaling_allowed, 60)

    def test_cooldown_is_in_progress_scaling_unfinished(self):
        previous_meta = {'scaling_in_progress': True}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertRaises(resource.NoActionRequired, self.group._check_scaling_allowed, 60)

    def test_scaling_not_in_progress_legacy(self):
        awhile_ago = timeutils.utcnow() - datetime.timedelta(seconds=100)
        previous_meta = {'cooldown': {awhile_ago.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertIsNone(self.group._check_scaling_allowed(60))

    def test_scaling_not_in_progress(self):
        awhile_after = timeutils.utcnow() + datetime.timedelta(seconds=60)
        previous_meta = {'cooldown_end': {awhile_after.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
        timeutils.set_time_override()
        timeutils.advance_time_seconds(100)
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertIsNone(self.group._check_scaling_allowed(60))
        timeutils.clear_time_override()

    def test_scaling_policy_cooldown_zero(self):
        now = timeutils.utcnow()
        previous_meta = {'cooldown_end': {now.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertIsNone(self.group._check_scaling_allowed(60))

    def test_scaling_policy_cooldown_none(self):
        now = timeutils.utcnow()
        previous_meta = {'cooldown_end': {now.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        self.assertIsNone(self.group._check_scaling_allowed(None))

    def test_metadata_is_written(self):
        nowish = timeutils.utcnow()
        reason = 'cool as'
        meta_set = self.patchobject(self.group, 'metadata_set')
        self.patchobject(timeutils, 'utcnow', return_value=nowish)
        self.group._finished_scaling(60, reason)
        cooldown_end = nowish + datetime.timedelta(seconds=60)
        meta_set.assert_called_once_with({'cooldown_end': {cooldown_end.isoformat(): reason}, 'scaling_in_progress': False})

    def test_metadata_is_written_update(self):
        nowish = timeutils.utcnow()
        reason = 'cool as'
        prev_cooldown_end = nowish + datetime.timedelta(seconds=100)
        previous_meta = {'cooldown_end': {prev_cooldown_end.isoformat(): 'ChangeInCapacity : 1'}}
        self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
        meta_set = self.patchobject(self.group, 'metadata_set')
        self.patchobject(timeutils, 'utcnow', return_value=nowish)
        self.group._finished_scaling(60, reason)
        meta_set.assert_called_once_with({'cooldown_end': {prev_cooldown_end.isoformat(): reason}, 'scaling_in_progress': False})