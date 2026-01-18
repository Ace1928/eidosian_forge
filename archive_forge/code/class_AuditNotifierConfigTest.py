from unittest import mock
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
class AuditNotifierConfigTest(base.BaseAuditMiddlewareTest):

    def test_conf_middleware_log_and_default_as_messaging(self):
        self.cfg.config(driver='log', group='audit_middleware_notifications')
        app = self.create_simple_app()
        with mock.patch('oslo_messaging.notify._impl_log.LogDriver.notify', side_effect=Exception('error')) as driver:
            app.get('/foo/bar', extra_environ=self.get_environ_header())
            self.assertTrue(driver.called)

    def test_conf_middleware_log_and_oslo_msg_as_messaging(self):
        self.cfg.config(driver=['messaging'], group='oslo_messaging_notifications')
        self.cfg.config(driver='log', group='audit_middleware_notifications')
        app = self.create_simple_app()
        with mock.patch('oslo_messaging.notify._impl_log.LogDriver.notify', side_effect=Exception('error')) as driver:
            app.get('/foo/bar', extra_environ=self.get_environ_header())
            self.assertTrue(driver.called)

    def test_conf_middleware_messaging_and_oslo_msg_as_log(self):
        self.cfg.config(driver=['log'], group='oslo_messaging_notifications')
        self.cfg.config(driver='messaging', group='audit_middleware_notifications')
        app = self.create_simple_app()
        with mock.patch('oslo_messaging.notify.messaging.MessagingDriver.notify', side_effect=Exception('error')) as driver:
            app.get('/foo/bar', extra_environ=self.get_environ_header())
            self.assertTrue(driver.called)

    def test_with_no_middleware_notification_conf(self):
        self.cfg.config(driver=['messaging'], group='oslo_messaging_notifications')
        self.cfg.config(driver=None, group='audit_middleware_notifications')
        app = self.create_simple_app()
        with mock.patch('oslo_messaging.notify.messaging.MessagingDriver.notify', side_effect=Exception('error')) as driver:
            app.get('/foo/bar', extra_environ=self.get_environ_header())
            self.assertTrue(driver.called)

    @mock.patch('oslo_messaging.get_notification_transport')
    def test_conf_middleware_messaging_and_transport_set(self, m):
        transport_url = 'rabbit://me:passwd@host:5672/virtual_host'
        self.cfg.config(driver='messaging', transport_url=transport_url, group='audit_middleware_notifications')
        self.create_simple_middleware()
        self.assertTrue(m.called)
        self.assertEqual(transport_url, m.call_args_list[0][1]['url'])

    def test_do_not_use_oslo_messaging(self):
        self.cfg.config(use_oslo_messaging=False, group='audit_middleware_notifications')
        audit_middleware = self.create_simple_middleware()
        self.assertIsInstance(audit_middleware._notifier, audit._notifier._LogNotifier)