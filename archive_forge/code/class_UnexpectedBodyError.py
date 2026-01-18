from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class UnexpectedBodyError(Error):
    """Exception raised by RequestMockBuilder on unexpected bodies."""

    def __init__(self, expected, provided):
        """Constructor for an UnexpectedMethodError."""
        super(UnexpectedBodyError, self).__init__('Expected: [%s] - Provided: [%s]' % (expected, provided))