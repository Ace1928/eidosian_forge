import socket
import unittest
import httplib2
from six.moves import http_client
from mock import patch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class RaisesExceptionOnLen(object):
    """Supports length property but raises if __len__ is used."""

    def __len__(self):
        raise Exception('len() called unnecessarily')

    def length(self):
        return 1