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
class ProvDerivation(ProvRelation):
    """Provenance Derivation relationship."""
    FORMAL_ATTRIBUTES = (PROV_ATTR_GENERATED_ENTITY, PROV_ATTR_USED_ENTITY, PROV_ATTR_ACTIVITY, PROV_ATTR_GENERATION, PROV_ATTR_USAGE)
    _prov_type = PROV_DERIVATION