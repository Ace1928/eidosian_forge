from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
class ApiResponse(Response):
    """
    Returns copy of the requests.Response object provides additional helper
    routines
        1. obj: returns dictionary of Avi Object
    """

    def __init__(self, rsp):
        super(ApiResponse, self).__init__()
        for k, v in list(rsp.__dict__.items()):
            setattr(self, k, v)

    def json(self):
        """
        Extends the session default json interface to handle special errors
        and raise Exceptions
        returns the Avi object as a dictionary from rsp.text
        """
        if self.status_code in (200, 201):
            if not self.text:
                return None
            return super(ApiResponse, self).json()
        elif self.status_code == 204:
            return None
        elif self.status_code == 404:
            raise ObjectNotFound('HTTP Error: %s Error Msg %s' % (self.status_code, self.text), self)
        elif self.status_code >= 500:
            raise AviServerError('HTTP Error: %s Error Msg %s' % (self.status_code, self.text), self)
        else:
            raise APIError('HTTP Error: %s Error Msg %s' % (self.status_code, self.text), self)

    def count(self):
        """
        return the number of objects in the collection response. If it is not
        a collection response then it would simply return 1.
        """
        obj = self.json()
        if 'count' in obj:
            return obj['count']
        return 1

    @staticmethod
    def to_avi_response(resp):
        if type(resp) is Response:
            return ApiResponse(resp)
        return resp