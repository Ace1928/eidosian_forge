import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
@classmethod
def _normalize_request_dict(cls, data):
    """
        This class method recurses through request data dictionary and removes
        any default values.

        :type data: dict
        :param data: Specifies request parameters with default values to be removed.
        """
    for item in list(data.keys()):
        if isinstance(data[item], dict):
            cls._normalize_request_dict(data[item])
        if data[item] in (None, {}):
            del data[item]