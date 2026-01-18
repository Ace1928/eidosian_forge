import abc
import errno
import functools
import os
import re
import signal
import struct
import subprocess
import sys
import time
from eventlet.green import socket
import eventlet.greenio
import eventlet.wsgi
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from osprofiler import opts as profiler_opts
import routes.middleware
import webob.dec
import webob.exc
from webob import multidict
from glance.common import config
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
import glance.db
from glance import housekeeping
from glance import i18n
from glance.i18n import _, _LE, _LI, _LW
from glance import sqlite_migration
class JSONRequestDeserializer(object):
    valid_transfer_encoding = frozenset(['chunked', 'compress', 'deflate', 'gzip', 'identity'])
    httpverb_may_have_body = frozenset({'POST', 'PUT', 'PATCH'})

    @classmethod
    def is_valid_encoding(cls, request):
        request_encoding = request.headers.get('transfer-encoding', '').lower()
        return request_encoding in cls.valid_transfer_encoding

    @classmethod
    def is_valid_method(cls, request):
        return request.method.upper() in cls.httpverb_may_have_body

    def has_body(self, request):
        """
        Returns whether a Webob.Request object will possess an entity body.

        :param request:  Webob.Request object
        """
        if self.is_valid_encoding(request) and self.is_valid_method(request):
            request.is_body_readable = True
            return True
        if request.content_length is not None and request.content_length > 0:
            return True
        return False

    @staticmethod
    def _sanitizer(obj):
        """Sanitizer method that will be passed to jsonutils.loads."""
        return obj

    def from_json(self, datastring):
        try:
            jsondata = jsonutils.loads(datastring, object_hook=self._sanitizer)
            if not isinstance(jsondata, (dict, list)):
                msg = _('Unexpected body type. Expected list/dict.')
                raise webob.exc.HTTPBadRequest(explanation=msg)
            return jsondata
        except ValueError:
            msg = _('Malformed JSON in request body.')
            raise webob.exc.HTTPBadRequest(explanation=msg)

    def default(self, request):
        if self.has_body(request):
            return {'body': self.from_json(request.body)}
        else:
            return {}