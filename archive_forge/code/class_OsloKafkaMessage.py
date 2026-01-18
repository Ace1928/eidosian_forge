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
class OsloKafkaMessage(base.RpcIncomingMessage):

    def __init__(self, ctxt, message):
        super(OsloKafkaMessage, self).__init__(ctxt, message)

    def requeue(self):
        LOG.warning('requeue is not supported')

    def reply(self, reply=None, failure=None):
        LOG.warning('reply is not supported')

    def heartbeat(self):
        LOG.warning('heartbeat is not supported')