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
def _ensure_producer(self):
    if self.producer:
        return
    with self.producer_lock:
        if self.producer:
            return
        conf = {'bootstrap.servers': ','.join(self.hostaddrs), 'linger.ms': self.linger_ms, 'batch.num.messages': self.batch_size, 'compression.codec': self.compression_codec, 'security.protocol': self.security_protocol, 'sasl.mechanism': self.sasl_mechanism, 'sasl.username': self.username, 'sasl.password': self.password, 'ssl.ca.location': self.ssl_cafile, 'ssl.certificate.location': self.ssl_client_cert_file, 'ssl.key.location': self.ssl_client_key_file, 'ssl.key.password': self.ssl_client_key_password}
        self.producer = confluent_kafka.Producer(conf)