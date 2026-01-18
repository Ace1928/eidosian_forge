import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def HttpRequestSideEffect(self, responses=None):
    responses = [(response.info, response.content) for response in responses]

    def _side_effect(uri, **kwargs):
        body = kwargs['body']
        read_func = getattr(body, 'read', None)
        if read_func:
            body = read_func()
        self.assertEqual(int(kwargs['headers']['content-length']), len(body))
        return responses.pop(0)
    return _side_effect