from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def _conn_ready(self, conn):
    """
        There is a nice state diagram at the top of http_client.py.  It
        indicates that once the response headers have been read (which
        _mexe does before adding the connection to the pool), a
        response is attached to the connection, and it stays there
        until it's done reading.  This isn't entirely true: even after
        the client is done reading, the response may be closed, but
        not removed from the connection yet.

        This is ugly, reading a private instance variable, but the
        state we care about isn't available in any public methods.
        """
    if ON_APP_ENGINE:
        return False
    else:
        response = getattr(conn, '_HTTPConnection__response', None)
        return response is None or response.isclosed()