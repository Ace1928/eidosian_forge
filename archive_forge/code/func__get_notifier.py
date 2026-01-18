import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_notifier():
    """Return a notifier object.

    If _notifier is None it means that a notifier object has not been set.
    If _notifier is False it means that a notifier has previously failed to
    construct.
    Otherwise it is a constructed Notifier object.
    """
    global _notifier
    if _notifier is None:
        host = CONF.default_publisher_id or socket.gethostname()
        try:
            transport = oslo_messaging.get_notification_transport(CONF)
            _notifier = oslo_messaging.Notifier(transport, 'identity.%s' % host)
        except Exception:
            LOG.exception('Failed to construct notifier')
            _notifier = False
    return _notifier