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
class ProvMention(ProvSpecialization):
    """Provenance Mention relationship (specific Specialization)."""
    FORMAL_ATTRIBUTES = (PROV_ATTR_SPECIFIC_ENTITY, PROV_ATTR_GENERAL_ENTITY, PROV_ATTR_BUNDLE)
    _prov_type = PROV_MENTION