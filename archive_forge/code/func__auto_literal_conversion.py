from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
def _auto_literal_conversion(self, literal):
    if isinstance(literal, ProvRecord):
        literal = literal.identifier
    if isinstance(literal, str):
        return str(literal)
    elif isinstance(literal, QualifiedName):
        return self._bundle.valid_qualified_name(literal)
    elif isinstance(literal, Literal) and literal.has_no_langtag():
        if literal.datatype:
            value = parse_xsd_types(literal.value, literal.datatype)
        else:
            value = self._auto_literal_conversion(literal.value)
        if value is not None:
            return value
    return literal