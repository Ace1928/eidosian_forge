import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
class SwiftClientPluginTestCase(common.HeatTestCase):

    def setUp(self):
        super(SwiftClientPluginTestCase, self).setUp()
        self.swift_client = mock.Mock()
        self.context = utils.dummy_context()
        self.context.project_id = 'demo'
        c = self.context.clients
        self.swift_plugin = c.client_plugin('swift')
        self.swift_plugin.client = lambda: self.swift_client