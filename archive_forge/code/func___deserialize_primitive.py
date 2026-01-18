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
def __deserialize_primitive(self, data, klass):
    """
        Deserializes string to primitive type.

        :param data: str.
        :param klass: class literal.

        :return: int, long, float, str, bool.
        """
    try:
        return klass(data)
    except UnicodeEncodeError:
        return unicode(data)
    except TypeError:
        return data