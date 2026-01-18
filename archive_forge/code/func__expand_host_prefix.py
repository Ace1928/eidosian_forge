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
def _expand_host_prefix(self, parameters, operation_model):
    operation_endpoint = operation_model.endpoint
    if operation_endpoint is None or 'hostPrefix' not in operation_endpoint:
        return None
    host_prefix_expression = operation_endpoint['hostPrefix']
    input_members = operation_model.input_shape.members
    host_labels = [member for member, shape in input_members.items() if shape.serialization.get('hostLabel')]
    format_kwargs = {}
    bad_labels = []
    for name in host_labels:
        param = parameters[name]
        if not HOST_PREFIX_RE.match(param):
            bad_labels.append(name)
        format_kwargs[name] = param
    if bad_labels:
        raise ParamValidationError(report=f'Invalid value for parameter(s): {', '.join(bad_labels)}. Must contain only alphanumeric characters, hyphen, or period.')
    return host_prefix_expression.format(**format_kwargs)