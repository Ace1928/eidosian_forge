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
class RestJSONSerializer(BaseRestSerializer, JSONSerializer):

    def _serialize_empty_body(self):
        return b'{}'

    def _requires_empty_body(self, shape):
        """
        Serialize an empty JSON object whenever the shape has
        members not targeting a location.
        """
        for member, val in shape.members.items():
            if 'location' not in val.serialization:
                return True
        return False

    def _serialize_content_type(self, serialized, shape, shape_members):
        """Set Content-Type to application/json for all structured bodies."""
        payload = shape.serialization.get('payload')
        if self._has_streaming_payload(payload, shape_members):
            return
        has_body = serialized['body'] != b''
        has_content_type = has_header('Content-Type', serialized['headers'])
        if has_body and (not has_content_type):
            serialized['headers']['Content-Type'] = 'application/json'

    def _serialize_body_params(self, params, shape):
        serialized_body = self.MAP_TYPE()
        self._serialize(serialized_body, params, shape)
        return json.dumps(serialized_body).encode(self.DEFAULT_ENCODING)