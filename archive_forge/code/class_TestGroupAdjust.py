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
class TestGroupAdjust(common.HeatTestCase):

    def setUp(self):
        super(TestGroupAdjust, self).setUp()
        t = template_format.parse(as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.group = self.stack['WebServerGroup']
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        self.stub_SnapshotConstraint_validate()
        self.assertIsNone(self.group.validate())

    def test_scaling_policy_cooldown_toosoon(self):
        dont_call = self.patchobject(self.group, 'resize')
        self.patchobject(self.group, '_check_scaling_allowed', side_effect=resource.NoActionRequired)
        self.assertRaises(resource.NoActionRequired, self.group.adjust, 1)
        self.assertEqual([], dont_call.call_args_list)

    def test_scaling_same_capacity(self):
        """Don't resize when capacity is the same."""
        self.patchobject(grouputils, 'get_size', return_value=3)
        resize = self.patchobject(self.group, 'resize')
        finished_scaling = self.patchobject(self.group, '_finished_scaling')
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.assertRaises(resource.NoActionRequired, self.group.adjust, 3, adjustment_type='ExactCapacity')
        expected_notifies = []
        self.assertEqual(expected_notifies, notify.call_args_list)
        self.assertEqual(0, resize.call_count)
        self.assertEqual(0, finished_scaling.call_count)

    def test_scaling_update_in_progress(self):
        """Don't resize when update in progress"""
        self.group.state_set('UPDATE', 'IN_PROGRESS')
        resize = self.patchobject(self.group, 'resize')
        finished_scaling = self.patchobject(self.group, '_finished_scaling')
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.assertRaises(resource.NoActionRequired, self.group.adjust, 3, adjustment_type='ExactCapacity')
        expected_notifies = []
        self.assertEqual(expected_notifies, notify.call_args_list)
        self.assertEqual(0, resize.call_count)
        self.assertEqual(0, finished_scaling.call_count)

    def test_scale_up_min_adjustment(self):
        self.patchobject(grouputils, 'get_size', return_value=1)
        resize = self.patchobject(self.group, 'resize')
        finished_scaling = self.patchobject(self.group, '_finished_scaling')
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.patchobject(self.group, '_check_scaling_allowed')
        self.group.adjust(33, adjustment_type='PercentChangeInCapacity', min_adjustment_step=2)
        expected_notifies = [mock.call(capacity=1, suffix='start', adjustment_type='PercentChangeInCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=33, stack=self.group.stack), mock.call(capacity=3, suffix='end', adjustment_type='PercentChangeInCapacity', groupname=u'WebServerGroup', message=u'End resizing the group WebServerGroup', adjustment=33, stack=self.group.stack)]
        self.assertEqual(expected_notifies, notify.call_args_list)
        resize.assert_called_once_with(3)
        finished_scaling.assert_called_once_with(None, 'PercentChangeInCapacity : 33', size_changed=True)

    def test_scale_down_min_adjustment(self):
        self.patchobject(grouputils, 'get_size', return_value=5)
        resize = self.patchobject(self.group, 'resize')
        finished_scaling = self.patchobject(self.group, '_finished_scaling')
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.patchobject(self.group, '_check_scaling_allowed')
        self.group.adjust(-33, adjustment_type='PercentChangeInCapacity', min_adjustment_step=2)
        expected_notifies = [mock.call(capacity=5, suffix='start', adjustment_type='PercentChangeInCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=-33, stack=self.group.stack), mock.call(capacity=3, suffix='end', adjustment_type='PercentChangeInCapacity', groupname=u'WebServerGroup', message=u'End resizing the group WebServerGroup', adjustment=-33, stack=self.group.stack)]
        self.assertEqual(expected_notifies, notify.call_args_list)
        resize.assert_called_once_with(3)
        finished_scaling.assert_called_once_with(None, 'PercentChangeInCapacity : -33', size_changed=True)

    def test_scaling_policy_cooldown_ok(self):
        self.patchobject(grouputils, 'get_size', return_value=0)
        resize = self.patchobject(self.group, 'resize')
        finished_scaling = self.patchobject(self.group, '_finished_scaling')
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.patchobject(self.group, '_check_scaling_allowed')
        self.group.adjust(1)
        expected_notifies = [mock.call(capacity=0, suffix='start', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=1, stack=self.group.stack), mock.call(capacity=1, suffix='end', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'End resizing the group WebServerGroup', adjustment=1, stack=self.group.stack)]
        self.assertEqual(expected_notifies, notify.call_args_list)
        resize.assert_called_once_with(1)
        finished_scaling.assert_called_once_with(None, 'ChangeInCapacity : 1', size_changed=True)
        grouputils.get_size.assert_called_once_with(self.group)

    def test_scaling_policy_resize_fail(self):
        self.patchobject(grouputils, 'get_size', return_value=0)
        self.patchobject(self.group, 'resize', side_effect=ValueError('test error'))
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.patchobject(self.group, '_check_scaling_allowed')
        self.patchobject(self.group, '_finished_scaling')
        self.assertRaises(ValueError, self.group.adjust, 1)
        expected_notifies = [mock.call(capacity=0, suffix='start', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=1, stack=self.group.stack), mock.call(capacity=0, suffix='error', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'test error', adjustment=1, stack=self.group.stack)]
        self.assertEqual(expected_notifies, notify.call_args_list)
        grouputils.get_size.assert_called_with(self.group)

    def test_notification_send_if_resize_failed(self):
        """If resize failed, the capacity of group might have been changed"""
        self.patchobject(grouputils, 'get_size', side_effect=[3, 4])
        self.patchobject(self.group, 'resize', side_effect=ValueError('test error'))
        notify = self.patch('heat.engine.notification.autoscaling.send')
        self.patchobject(self.group, '_check_scaling_allowed')
        self.patchobject(self.group, '_finished_scaling')
        self.assertRaises(ValueError, self.group.adjust, 5, adjustment_type='ExactCapacity')
        expected_notifies = [mock.call(capacity=3, suffix='start', adjustment_type='ExactCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=5, stack=self.group.stack), mock.call(capacity=4, suffix='error', adjustment_type='ExactCapacity', groupname=u'WebServerGroup', message=u'test error', adjustment=5, stack=self.group.stack)]
        self.assertEqual(expected_notifies, notify.call_args_list)
        self.group.resize.assert_called_once_with(5)
        grouputils.get_size.assert_has_calls([mock.call(self.group), mock.call(self.group)])