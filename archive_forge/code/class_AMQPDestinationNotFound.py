import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
class AMQPDestinationNotFound(Exception):
    pass