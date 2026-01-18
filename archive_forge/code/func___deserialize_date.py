from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def __deserialize_date(self, string):
    """
        Deserializes string to date.

        :param string: str.
        :return: date.
        """
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string
    except ValueError:
        raise ApiException(status=0, reason='Failed to parse `{0}` into a date object'.format(string))