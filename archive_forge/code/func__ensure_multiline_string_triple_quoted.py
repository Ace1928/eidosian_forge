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
def _ensure_multiline_string_triple_quoted(value):
    s = str(value)
    s = s.replace('"', '\\"')
    if '\n' in s:
        return '"""%s"""' % s
    else:
        return '"%s"' % s