import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
class TestEventCallbacks(test_v3.RestfulTestCase):

    class FakeManager(object):

        def _project_deleted_callback(self, service, resource_type, operation, payload):
            """Used just for the callback interface."""

    def test_notification_received(self):
        callback = register_callback(CREATED_OPERATION, 'project')
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        self.assertTrue(callback.called)

    def test_notification_method_not_callable(self):
        fake_method = None
        self.assertRaises(TypeError, notifications.register_event_callback, UPDATED_OPERATION, 'project', [fake_method])

    def test_notification_event_not_valid(self):
        manager = self.FakeManager()
        self.assertRaises(ValueError, notifications.register_event_callback, uuid.uuid4().hex, 'project', manager._project_deleted_callback)

    def test_event_registration_for_unknown_resource_type(self):
        manager = self.FakeManager()
        notifications.register_event_callback(DELETED_OPERATION, uuid.uuid4().hex, manager._project_deleted_callback)
        resource_type = uuid.uuid4().hex
        notifications.register_event_callback(DELETED_OPERATION, resource_type, manager._project_deleted_callback)

    def test_provider_event_callback_subscription(self):
        callback_called = []

        @notifications.listener
        class Foo(object):

            def __init__(self):
                self.event_callbacks = {CREATED_OPERATION: {'project': self.foo_callback}}

            def foo_callback(self, service, resource_type, operation, payload):
                callback_called.append(True)
        Foo()
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        self.assertEqual([True], callback_called)

    def test_provider_event_callbacks_subscription(self):
        callback_called = []

        @notifications.listener
        class Foo(object):

            def __init__(self):
                self.event_callbacks = {CREATED_OPERATION: {'project': [self.callback_0, self.callback_1]}}

            def callback_0(self, service, resource_type, operation, payload):
                callback_called.append('cb0')

            def callback_1(self, service, resource_type, operation, payload):
                callback_called.append('cb1')
        Foo()
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        self.assertCountEqual(['cb1', 'cb0'], callback_called)

    def test_invalid_event_callbacks(self):

        @notifications.listener
        class Foo(object):

            def __init__(self):
                self.event_callbacks = 'bogus'
        self.assertRaises(AttributeError, Foo)

    def test_invalid_event_callbacks_event(self):

        @notifications.listener
        class Foo(object):

            def __init__(self):
                self.event_callbacks = {CREATED_OPERATION: 'bogus'}
        self.assertRaises(AttributeError, Foo)

    def test_using_an_unbound_method_as_a_callback_fails(self):

        @notifications.listener
        class Foo(object):

            def __init__(self):
                self.event_callbacks = {CREATED_OPERATION: {'project': Foo.callback}}

            def callback(self, service, resource_type, operation, payload):
                pass
        Foo()
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        self.assertRaises(TypeError, PROVIDERS.resource_api.create_project, project_ref['id'], project_ref)