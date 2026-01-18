import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def Unmock(self, suppress=False):
    self.__class__ = self.__orig_class
    for name, service_class in self.__real_service_classes.items():
        setattr(self.__client_class, name, service_class)
        delattr(self, service_class._NAME)
    self.__real_service_classes = {}
    del self._url
    del self._http
    self.__client_class.IncludeFields = self.__real_include_fields
    self.__real_include_fields = None
    requests = [(rq_rs.key, rq_rs.request) for rq_rs in self._request_responses]
    self._request_responses = []
    if requests and (not suppress) and (sys.exc_info()[1] is None):
        raise ExpectedRequestsException(requests)