from __future__ import absolute_import, unicode_literals
from collections import OrderedDict
import six
import yaml
from pybtex.database import Entry, Person
from pybtex.database.input import BaseParser

    SafeLoader that loads mappings as OrderedDicts.
    