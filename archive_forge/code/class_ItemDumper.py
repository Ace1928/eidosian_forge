from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class ItemDumper(yaml.SafeDumper):
    """For dumping validation.Items. Respects SortedDict key ordering."""

    def represent_mapping(self, tag, mapping, flow_style=None):
        if hasattr(mapping, 'ordered_items'):
            return super(ItemDumper, self).represent_mapping(tag, mapping.ordered_items(), flow_style=flow_style)
        return super(ItemDumper, self).represent_mapping(tag, mapping, flow_style=flow_style)