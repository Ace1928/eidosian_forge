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
class ProvExceptionInvalidQualifiedName(ProvException):
    """Exception for an invalid qualified identifier name."""
    qname = None
    'Intended qualified name.'

    def __init__(self, qname):
        """
        Constructor.

        :param qname: Invalid qualified name.
        """
        self.qname = qname

    def __str__(self):
        return 'Invalid Qualified Name: %s' % self.qname