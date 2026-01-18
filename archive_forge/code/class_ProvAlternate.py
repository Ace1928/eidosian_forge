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
class ProvAlternate(ProvRelation):
    """Provenance Alternate relationship."""
    FORMAL_ATTRIBUTES = (PROV_ATTR_ALTERNATE1, PROV_ATTR_ALTERNATE2)
    _prov_type = PROV_ALTERNATE