import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
class AsynchronousMessage(object):
    """A container for handling asynchronous LDAP responses.

    Some LDAP APIs, like `search_ext`, are asynchronous and return a message ID
    when the server successfully initiates the operation. Clients can use this
    message ID and the original connection to make the request to fetch the
    results using `result3`.

    This object holds the message ID, the original connection, and a callable
    weak reference Finalizer that cleans up context managers specific to the
    connection associated to the message ID.

    :param message_id: The message identifier (str).
    :param connection: The connection associated with the message identifier
                       (ldappool.StateConnector).

    The `clean` attribute is a callable that cleans up the context manager used
    to create or return the connection object (weakref.finalize).

    """

    def __init__(self, message_id, connection, context_manager):
        self.id = message_id
        self.connection = connection
        self.clean = weakref.finalize(self, self._cleanup_connection_context_manager, context_manager)

    def _cleanup_connection_context_manager(self, context_manager):
        context_manager.__exit__(None, None, None)