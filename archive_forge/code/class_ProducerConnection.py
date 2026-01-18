import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
class ProducerConnection(Connection):

    def __init__(self, conf, url):
        super(ProducerConnection, self).__init__(conf, url)
        self.batch_size = self.driver_conf.producer_batch_size
        self.linger_ms = self.driver_conf.producer_batch_timeout * 1000
        self.compression_codec = self.driver_conf.compression_codec
        self.producer = None
        self.producer_lock = threading.Lock()

    def _produce_message(self, topic, message, poll):
        if poll:
            self.producer.poll(poll)
        try:
            self.producer.produce(topic, message)
        except KafkaException as e:
            self.producer.poll(0)
            raise e
        except BufferError as e:
            raise e
        self.producer.poll(0)

    def notify_send(self, topic, ctxt, msg, retry):
        """Send messages to Kafka broker.

        :param topic: String of the topic
        :param ctxt: context for the messages
        :param msg: messages for publishing
        :param retry: the number of retry
        """
        retry = retry if retry >= 0 else None
        message = pack_message(ctxt, msg)
        message = jsonutils.dumps(message).encode('utf-8')
        try:
            self._ensure_producer()
            poll = 0
            while True:
                try:
                    if eventletutils.is_monkey_patched('thread'):
                        return tpool.execute(self._produce_message, topic, message, poll)
                    return self._produce_message(topic, message, poll)
                except KafkaException as e:
                    LOG.error('Produce message failed: %s' % str(e))
                    break
                except BufferError:
                    LOG.debug('Produce message queue full, waiting for deliveries')
                    poll = 0.5
        except Exception:
            self._close_producer()
            raise

    def close(self):
        self._close_producer()

    def _close_producer(self):
        with self.producer_lock:
            if self.producer:
                try:
                    self.producer.flush()
                except KafkaException:
                    LOG.error('Flush error during producer close')
                self.producer = None

    def _ensure_producer(self):
        if self.producer:
            return
        with self.producer_lock:
            if self.producer:
                return
            conf = {'bootstrap.servers': ','.join(self.hostaddrs), 'linger.ms': self.linger_ms, 'batch.num.messages': self.batch_size, 'compression.codec': self.compression_codec, 'security.protocol': self.security_protocol, 'sasl.mechanism': self.sasl_mechanism, 'sasl.username': self.username, 'sasl.password': self.password, 'ssl.ca.location': self.ssl_cafile, 'ssl.certificate.location': self.ssl_client_cert_file, 'ssl.key.location': self.ssl_client_key_file, 'ssl.key.password': self.ssl_client_key_password}
            self.producer = confluent_kafka.Producer(conf)