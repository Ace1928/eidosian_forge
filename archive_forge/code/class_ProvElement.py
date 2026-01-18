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
class ProvElement(ProvRecord):
    """Provenance Element (nodes in the provenance graph)."""

    def __init__(self, bundle, identifier, attributes=None):
        if identifier is None:
            raise ProvElementIdentifierRequired()
        super(ProvElement, self).__init__(bundle, identifier, attributes)

    def is_element(self):
        """
        True, if the record is an element, False otherwise.

        :return: bool
        """
        return True

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._identifier)