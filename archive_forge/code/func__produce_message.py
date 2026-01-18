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