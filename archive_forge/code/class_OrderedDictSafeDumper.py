from __future__ import unicode_literals
from collections import OrderedDict
import yaml
from pybtex.database.output import BaseWriter
class OrderedDictSafeDumper(yaml.SafeDumper):
    """
    SafeDumper that dumps OrderedDicts preserving the order.
    """

    def represent_odict(self, data):
        return self.represent_mapping(u'tag:yaml.org,2002:map', data.items())