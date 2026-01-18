from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def _get_file_segments(self, endpoint, filename, file_size, segment_size):
    segments = collections.OrderedDict()
    for index, offset in enumerate(range(0, file_size, segment_size)):
        remaining = file_size - index * segment_size
        segment = _utils.FileSegment(filename, offset, segment_size if segment_size < remaining else remaining)
        name = '{endpoint}/{index:0>6}'.format(endpoint=endpoint, index=index)
        segments[name] = segment
    return segments