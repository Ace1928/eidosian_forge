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
def __deserialize_model(self, data, klass):
    """
        Deserializes list or dict to model.

        :param data: dict, list.
        :param klass: class literal.
        :return: model object.
        """
    if not klass.swagger_types and (not hasattr(klass, 'get_real_child_model')):
        return data
    kwargs = {}
    if klass.swagger_types is not None:
        for attr, attr_type in iteritems(klass.swagger_types):
            if data is not None and klass.attribute_map[attr] in data and isinstance(data, (list, dict)):
                value = data[klass.attribute_map[attr]]
                kwargs[attr] = self.__deserialize(value, attr_type)
    instance = klass(**kwargs)
    if hasattr(instance, 'get_real_child_model'):
        klass_name = instance.get_real_child_model(data)
        if klass_name:
            instance = self.__deserialize(data, klass_name)
    return instance