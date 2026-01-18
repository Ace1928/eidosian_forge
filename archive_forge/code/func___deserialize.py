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
def __deserialize(self, data, klass):
    """
        Deserializes dict, list, str into an object.

        :param data: dict, list or str.
        :param klass: class literal, or string of class name.

        :return: object.
        """
    if data is None:
        return None
    if type(klass) == str:
        if klass.startswith('list['):
            sub_kls = re.match('list\\[(.*)\\]', klass).group(1)
            return [self.__deserialize(sub_data, sub_kls) for sub_data in data]
        if klass.startswith('dict('):
            sub_kls = re.match('dict\\(([^,]*), (.*)\\)', klass).group(2)
            return {k: self.__deserialize(v, sub_kls) for k, v in iteritems(data)}
        if klass in self.NATIVE_TYPES_MAPPING:
            klass = self.NATIVE_TYPES_MAPPING[klass]
        else:
            klass = getattr(models, klass)
    if klass in self.PRIMITIVE_TYPES:
        return self.__deserialize_primitive(data, klass)
    elif klass == object:
        return self.__deserialize_object(data)
    elif klass == date:
        return self.__deserialize_date(data)
    elif klass == datetime:
        return self.__deserialize_datatime(data)
    else:
        return self.__deserialize_model(data, klass)