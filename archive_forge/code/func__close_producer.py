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
def _close_producer(self):
    with self.producer_lock:
        if self.producer:
            try:
                self.producer.flush()
            except KafkaException:
                LOG.error('Flush error during producer close')
            self.producer = None