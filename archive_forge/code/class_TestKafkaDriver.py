import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
class TestKafkaDriver(test_utils.BaseTestCase):
    """Unit Test cases to test the kafka driver
    """

    def setUp(self):
        super(TestKafkaDriver, self).setUp()
        self.messaging_conf.transport_url = 'kafka:/'
        transport = oslo_messaging.get_notification_transport(self.conf)
        self.driver = transport._driver

    def test_send(self):
        target = oslo_messaging.Target(topic='topic_test')
        self.assertRaises(NotImplementedError, self.driver.send, target, {}, {})

    def test_send_notification(self):
        target = oslo_messaging.Target(topic='topic_test')
        with mock.patch('confluent_kafka.Producer') as producer:
            self.driver.send_notification(target, {}, {'payload': ['test_1']}, None, retry=3)
            producer.assert_called_once_with({'bootstrap.servers': '', 'linger.ms': mock.ANY, 'batch.num.messages': mock.ANY, 'compression.codec': 'none', 'security.protocol': 'PLAINTEXT', 'sasl.mechanism': 'PLAIN', 'sasl.username': mock.ANY, 'sasl.password': mock.ANY, 'ssl.ca.location': '', 'ssl.certificate.location': '', 'ssl.key.location': '', 'ssl.key.password': ''})

    def test_send_notification_retries_on_buffer_error(self):
        target = oslo_messaging.Target(topic='topic_test')
        with mock.patch('confluent_kafka.Producer') as producer:
            fake_producer = mock.MagicMock()
            fake_producer.produce = mock.Mock(side_effect=[BufferError, BufferError, None])
            producer.return_value = fake_producer
            self.driver.send_notification(target, {}, {'payload': ['test_1']}, None, retry=3)
            assert fake_producer.produce.call_count == 3

    def test_send_notification_stops_on_kafka_error(self):
        target = oslo_messaging.Target(topic='topic_test')
        with mock.patch('confluent_kafka.Producer') as producer:
            fake_producer = mock.MagicMock()
            fake_producer.produce = mock.Mock(side_effect=[KafkaException, None])
            producer.return_value = fake_producer
            self.driver.send_notification(target, {}, {'payload': ['test_1']}, None, retry=3)
            assert fake_producer.produce.call_count == 1

    def test_listen(self):
        target = oslo_messaging.Target(topic='topic_test')
        self.assertRaises(NotImplementedError, self.driver.listen, target, None, None)

    def test_listen_for_notifications(self):
        targets_and_priorities = [(oslo_messaging.Target(topic='topic_test_1'), 'sample')]
        with mock.patch('confluent_kafka.Consumer') as consumer:
            self.driver.listen_for_notifications(targets_and_priorities, 'kafka_test', 1000, 10)
            consumer.assert_called_once_with({'bootstrap.servers': '', 'enable.partition.eof': False, 'group.id': 'kafka_test', 'enable.auto.commit': mock.ANY, 'max.partition.fetch.bytes': mock.ANY, 'security.protocol': 'PLAINTEXT', 'sasl.mechanism': 'PLAIN', 'sasl.username': mock.ANY, 'sasl.password': mock.ANY, 'ssl.ca.location': '', 'ssl.certificate.location': '', 'ssl.key.location': '', 'ssl.key.password': '', 'default.topic.config': {'auto.offset.reset': 'latest'}})

    def test_cleanup(self):
        listeners = [mock.MagicMock(), mock.MagicMock()]
        self.driver.listeners.extend(listeners)
        self.driver.cleanup()
        for listener in listeners:
            listener.close.assert_called_once_with()