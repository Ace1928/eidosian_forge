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
class AssignedPartition(object):
    """This class is used by the ConsumerConnection to track the
    assigned partitions.
    """

    def __init__(self, topic, partition):
        super(AssignedPartition, self).__init__()
        self.topic = topic
        self.partition = partition
        self.skey = '%s %d' % (self.topic, self.partition)

    def to_dict(self):
        return {'topic': self.topic, 'partition': self.partition}