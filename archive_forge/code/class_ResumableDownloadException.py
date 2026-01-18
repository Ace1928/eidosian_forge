import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class ResumableDownloadException(Exception):
    """
    Exception raised for various resumable download problems.

    self.disposition is of type ResumableTransferDisposition.
    """

    def __init__(self, message, disposition):
        super(ResumableDownloadException, self).__init__(message, disposition)
        self.message = message
        self.disposition = disposition

    def __repr__(self):
        return 'ResumableDownloadException("%s", %s)' % (self.message, self.disposition)