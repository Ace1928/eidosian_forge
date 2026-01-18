from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
class ResourceNameTranslatorWrapper(object):
    """Data wrapper for resource objects returned by get_resource()."""

    def __init__(self, resource):
        self._resource = resource

    @property
    def asset_inventory_type(self):
        return self._resource.asset_inventory_type

    @property
    def krm_kind(self):
        return KrmKind(self._resource.krm_group, self._resource.krm_kind)

    @property
    def collection_name(self):
        return self._resource.get_full_collection_name()

    @property
    def resource_data(self):
        return self._resource