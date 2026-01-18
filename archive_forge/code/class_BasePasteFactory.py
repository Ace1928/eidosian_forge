import abc
import errno
import os
import signal
import sys
import time
import eventlet
from eventlet.green import socket
from eventlet.green import ssl
import eventlet.greenio
import eventlet.wsgi
import functools
from oslo_concurrency import processutils
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from paste.deploy import loadwsgi
from routes import middleware
import webob.dec
import webob.exc
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common.i18n import _
from heat.common import serializers
class BasePasteFactory(object, metaclass=abc.ABCMeta):
    """A base class for paste app and filter factories.

    Sub-classes must override the KEY class attribute and provide
    a __call__ method.
    """
    KEY = None

    def __init__(self, conf):
        self.conf = conf

    @abc.abstractmethod
    def __call__(self, global_conf, **local_conf):
        return

    def _import_factory(self, local_conf):
        """Import an app/filter class.

        Lookup the KEY from the PasteDeploy local conf and import the
        class named there. This class can then be used as an app or
        filter factory.

        Note we support the <module>:<class> format.

        Note also that if you do e.g.

          key =
              value

        then ConfigParser returns a value with a leading newline, so
        we strip() the value before using it.
        """
        class_name = local_conf[self.KEY].replace(':', '.').strip()
        return importutils.import_class(class_name)