import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
class ExpectedRequestsException(Error):

    def __init__(self, expected_calls):
        msg = 'expected:\n'
        for key, request in expected_calls:
            msg += '{key}({request})\n'.format(key=key, request=encoding.MessageToRepr(request, multiline=True))
        super(ExpectedRequestsException, self).__init__(msg)