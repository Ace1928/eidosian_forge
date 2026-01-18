import os
import time
import getpass
import json
import requests
from urllib.parse import urlparse
from .select import Select
from .help import Inspector, GraphData, PaintGraph, _DRAW_GRAPHS
from .manage import GlobalManager
from .uriutil import join_uri, file_path, uri_last
from .jsonutil import csv_to_json
from .errors import is_xnat_error
from .errors import catch_error
from .array import ArrayData
from .xpath_store import XpathStore
from . import xpass
def _get_entry_point(self):
    if self._entry is None:
        self._entry = '/REST'
        try:
            ans = self._exec('/data/JSESSION', force_preemptive_auth=True)
            self._jsession = 'JSESSIONID=' + str(ans)
            self._entry = '/data'
            if is_xnat_error(self._jsession):
                catch_error(self._jsession)
        except Exception as e:
            if '/data/JSESSION' not in str(e):
                raise e
    return self._entry