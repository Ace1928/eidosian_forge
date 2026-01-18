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
class ProvRelation(ProvRecord):
    """Provenance Relationship (edge between nodes)."""

    def is_relation(self):
        """
        True, if the record is a relation, False otherwise.

        :return: bool
        """
        return True

    def __repr__(self):
        identifier = ' %s' % self._identifier if self._identifier else ''
        element_1, element_2 = [qname for _, qname in self.formal_attributes[:2]]
        return '<%s:%s (%s, %s)>' % (self.__class__.__name__, identifier, element_1, element_2)