import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
class UnexpectedRequestException(Error):

    def __init__(self, received_call, expected_call):
        expected_key, expected_request = expected_call
        received_key, received_request = received_call
        expected_repr = encoding.MessageToRepr(expected_request, multiline=True)
        received_repr = encoding.MessageToRepr(received_request, multiline=True)
        expected_lines = expected_repr.splitlines()
        received_lines = received_repr.splitlines()
        diff_lines = difflib.unified_diff(expected_lines, received_lines)
        diff = '\n'.join(diff_lines)
        if expected_key != received_key:
            msg = '\n'.join(('expected: {expected_key}({expected_request})', 'received: {received_key}({received_request})', '')).format(expected_key=expected_key, expected_request=expected_repr, received_key=received_key, received_request=received_repr)
            super(UnexpectedRequestException, self).__init__(msg)
        else:
            msg = '\n'.join(('for request to {key},', 'expected: {expected_request}', 'received: {received_request}', 'diff: {diff}', '')).format(key=expected_key, expected_request=expected_repr, received_request=received_repr, diff=diff)
            super(UnexpectedRequestException, self).__init__(msg)