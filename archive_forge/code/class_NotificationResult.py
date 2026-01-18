import itertools
import logging
import operator
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
class NotificationResult(object):
    HANDLED = 'handled'
    REQUEUE = 'requeue'