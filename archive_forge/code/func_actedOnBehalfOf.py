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
def actedOnBehalfOf(self, responsible, activity=None, attributes=None):
    """
        Creates a new delegation record on behalf of this agent.

        :param responsible: Agent the responsibility is delegated to.
        :param activity: Optionally extra activity to state qualified delegation
            internally (default: None).
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    self._bundle.delegation(self, responsible, activity, other_attributes=attributes)
    return self