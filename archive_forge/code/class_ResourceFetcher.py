import json
import logging
import os
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.utils import misc
class ResourceFetcher(object):

    def __init__(self):
        self._db_handle = None
        self._url_handle = None

    @property
    def db_handle(self):
        if self._db_handle is None:
            self._db_handle = DB()
        return self._db_handle

    @property
    def url_handle(self):
        if self._url_handle is None:
            self._url_handle = UrlCaller()
        return self._url_handle