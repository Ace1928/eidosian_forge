from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
class TestInstanceHaNotifications(TestInstanceHaProxy):

    def test_notifications(self):
        self.verify_list(self.proxy.notifications, notification.Notification)

    def test_notification_get(self):
        self.verify_get(self.proxy.get_notification, notification.Notification)

    def test_notification_create(self):
        self.verify_create(self.proxy.create_notification, notification.Notification)