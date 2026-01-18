import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
class _GeneratorWithMeta(_RequestIdMixin):

    def __init__(self, paginate_func, collection, path, **params):
        self.paginate_func = paginate_func
        self.collection = collection
        self.path = path
        self.params = params
        self.generator = None
        self._request_ids_setup()

    def _paginate(self):
        for r in self.paginate_func(self.collection, self.path, **self.params):
            yield (r, r.request_ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if not self.generator:
            self.generator = self._paginate()
        try:
            obj, req_id = next(self.generator)
            self._append_request_ids(req_id)
        except StopIteration:
            raise StopIteration()
        return obj