import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def _ingest_parameter_spec(self, parameters):
    return {name: ParameterDefinition(name, spec['type'], spec.get('documentation'), spec.get('builtIn'), spec.get('default'), spec.get('required'), spec.get('deprecated')) for name, spec in parameters.items()}