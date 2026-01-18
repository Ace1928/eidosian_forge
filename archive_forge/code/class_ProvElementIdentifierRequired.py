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
class ProvElementIdentifierRequired(ProvException):
    """Exception for a missing element identifier."""

    def __str__(self):
        return 'An identifier is missing. All PROV elements require a valid identifier.'