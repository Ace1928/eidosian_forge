import kombu
from oslo_config import cfg
from oslo_messaging._drivers import common
from oslo_messaging import transport
import requests
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class NotificationTest(functional_base.FunctionalTestsBase):
    basic_template = '\nheat_template_version: 2013-05-23\nresources:\n  random1:\n    type: OS::Heat::RandomString\n'
    update_basic_template = '\nheat_template_version: 2013-05-23\nresources:\n  random1:\n    type: OS::Heat::RandomString\n  random2:\n    type: OS::Heat::RandomString\n'
    asg_template = "\nheat_template_version: 2013-05-23\nresources:\n  asg:\n    type: OS::Heat::AutoScalingGroup\n    properties:\n      resource:\n        type: OS::Heat::RandomString\n      min_size: 1\n      desired_capacity: 2\n      max_size: 3\n\n  scale_up_policy:\n    type: OS::Heat::ScalingPolicy\n    properties:\n      adjustment_type: change_in_capacity\n      auto_scaling_group_id: {get_resource: asg}\n      cooldown: 0\n      scaling_adjustment: 1\n\n  scale_down_policy:\n    type: OS::Heat::ScalingPolicy\n    properties:\n      adjustment_type: change_in_capacity\n      auto_scaling_group_id: {get_resource: asg}\n      cooldown: 0\n      scaling_adjustment: '-1'\n\noutputs:\n  scale_up_url:\n    value: {get_attr: [scale_up_policy, alarm_url]}\n  scale_dn_url:\n    value: {get_attr: [scale_down_policy, alarm_url]}\n"

    def setUp(self):
        super(NotificationTest, self).setUp()
        self.exchange = kombu.Exchange('heat', 'topic', durable=False)
        queue = kombu.Queue(exchange=self.exchange, routing_key='notifications.info', exclusive=True)
        self.conn = kombu.Connection(get_url(transport.get_transport(cfg.CONF).conf))
        self.ch = self.conn.channel()
        self.queue = queue(self.ch)
        self.queue.declare()

    def consume_events(self, handler, count):
        self.conn.drain_events()
        return len(handler.notifications) == count

    def test_basic_notifications(self):
        stack_identifier = self.stack_create(template=self.basic_template, enable_cleanup=False)
        self.update_stack(stack_identifier, template=self.update_basic_template)
        self.stack_suspend(stack_identifier)
        self.stack_resume(stack_identifier)
        self._stack_delete(stack_identifier)
        handler = NotificationHandler(stack_identifier.split('/')[0])
        with self.conn.Consumer(self.queue, callbacks=[handler.process_message], auto_declare=False):
            try:
                while True:
                    self.conn.drain_events(timeout=1)
            except Exception:
                pass
        for n in BASIC_NOTIFICATIONS:
            self.assertIn(n, handler.notifications)

    def test_asg_notifications(self):
        stack_identifier = self.stack_create(template=self.asg_template)
        for output in self.client.stacks.get(stack_identifier).outputs:
            if output['output_key'] == 'scale_dn_url':
                scale_down_url = output['output_value']
            else:
                scale_up_url = output['output_value']
        notifications = []
        handler = NotificationHandler(stack_identifier.split('/')[0], ASG_NOTIFICATIONS)
        with self.conn.Consumer(self.queue, callbacks=[handler.process_message], auto_declare=False):
            requests.post(scale_up_url, verify=self.verify_cert)
            self.assertTrue(test.call_until_true(20, 0, self.consume_events, handler, 2))
            notifications += handler.notifications
            handler.clear()
            requests.post(scale_down_url, verify=self.verify_cert)
            self.assertTrue(test.call_until_true(20, 0, self.consume_events, handler, 2))
            notifications += handler.notifications
        self.assertEqual(2, notifications.count(ASG_NOTIFICATIONS[0]))
        self.assertEqual(2, notifications.count(ASG_NOTIFICATIONS[1]))