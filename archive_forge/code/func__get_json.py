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
def _get_json(self, uri):
    """ Specific Interface._exec method to retrieve data.
            It forces the data format to csv and then puts it back to a
            json-like format.

            Parameters
            ----------
            uri: string
                URI of the resource to be accessed. e.g. /REST/projects

            Returns
            -------
            List of dicts containing the results
        """
    if 'format=json' in uri:
        uri = uri.replace('format=json', 'format=csv')
    elif '?' in uri:
        uri += '&format=csv'
    else:
        uri += '?format=csv'
    content = self._exec(uri, 'GET')
    if is_xnat_error(content):
        catch_error(content)
    json_content = csv_to_json(content)
    base_uri = uri.split('?')[0]
    if uri_last(base_uri) == 'files':
        for element in json_content:
            element['path'] = file_path(element['URI'])
    return json_content