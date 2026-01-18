import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
class _RequestHistoryTracker(object):

    def __init__(self):
        self.request_history = []

    def _add_to_history(self, request):
        self.request_history.append(request)

    @property
    def last_request(self):
        """Retrieve the latest request sent"""
        try:
            return self.request_history[-1]
        except IndexError:
            return None

    @property
    def called(self):
        return self.call_count > 0

    @property
    def called_once(self):
        return self.call_count == 1

    @property
    def call_count(self):
        return len(self.request_history)

    def reset(self):
        self.request_history = []