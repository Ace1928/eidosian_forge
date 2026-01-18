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
class DefaultMethodController(object):
    """Controller that handles the OPTIONS request method.

    This controller handles the OPTIONS request method and any of the HTTP
    methods that are not explicitly implemented by the application.
    """

    def options(self, req, allowed_methods, *args, **kwargs):
        """Return a response that includes the 'Allow' header.

        Return a response that includes the 'Allow' header listing the methods
        that are implemented. A 204 status code is used for this response.
        """
        raise webob.exc.HTTPNoContent(headers=[('Allow', allowed_methods)])

    def reject(self, req, allowed_methods, *args, **kwargs):
        """Return a 405 method not allowed error.

        As a convenience, the 'Allow' header with the list of implemented
        methods is included in the response as well.
        """
        raise webob.exc.HTTPMethodNotAllowed(headers=[('Allow', allowed_methods)])