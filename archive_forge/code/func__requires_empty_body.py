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
def _requires_empty_body(self, shape):
    """
        Serialize an empty JSON object whenever the shape has
        members not targeting a location.
        """
    for member, val in shape.members.items():
        if 'location' not in val.serialization:
            return True
    return False