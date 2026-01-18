import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
class EC2Serializer(QuerySerializer):
    """EC2 specific customizations to the query protocol serializers.

    The EC2 model is almost, but not exactly, similar to the query protocol
    serializer.  This class encapsulates those differences.  The model
    will have be marked with a ``protocol`` of ``ec2``, so you don't need
    to worry about wiring this class up correctly.

    """

    def _get_serialized_name(self, shape, default_name):
        if 'queryName' in shape.serialization:
            return shape.serialization['queryName']
        elif 'name' in shape.serialization:
            name = shape.serialization['name']
            return name[0].upper() + name[1:]
        else:
            return default_name

    def _serialize_type_list(self, serialized, value, shape, prefix=''):
        for i, element in enumerate(value, 1):
            element_prefix = f'{prefix}.{i}'
            element_shape = shape.member
            self._serialize(serialized, element, element_shape, element_prefix)