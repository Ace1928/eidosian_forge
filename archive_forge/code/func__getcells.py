import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
def _getcells(self, cols):
    """ Gets multiple properties of the element resource.
        """
    p_uri = uri_parent(self._uri)
    id_head = schema.json[self._urt][0]
    lbl_head = schema.json[self._urt][1]
    filters = {}
    columns = set([col for col in cols if col not in schema.json[self._urt] or col != 'URI'] + schema.json[self._urt])
    get_id = p_uri + '?format=json&columns=%s' % ','.join(columns)
    for pattern in self._intf._struct.keys():
        if fnmatch(uri_segment(self._uri.split(self._intf._get_entry_point(), 1)[1], -2), pattern):
            reg_pat = self._intf._struct[pattern]
            filters.setdefault('xsiType', set()).add(reg_pat)
    if filters:
        get_id += '&' + '&'.join(('%s=%s' % (item[0], item[1]) if isinstance(item[1], str) else '%s=%s' % (item[0], ','.join([val for val in item[1]])) for item in filters.items()))
    for res in self._intf._get_json(get_id):
        if self._urn in [res.get(id_head), res.get(lbl_head)]:
            if len(cols) == 1:
                return res.get(cols[0])
            else:
                return get_selection(res, cols)[0]