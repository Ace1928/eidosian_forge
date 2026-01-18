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
def __deserialize_datatime(self, string):
    """
        Deserializes string to datetime.

        The string should be in iso8601 datetime format.

        :param string: str.
        :return: datetime.
        """
    try:
        from dateutil.parser import parse
        return parse(string)
    except ImportError:
        return string
    except ValueError:
        raise ApiException(status=0, reason='Failed to parse `{0}` into a datetime object'.format(string))