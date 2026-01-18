import logging
from functools import partial
from .action import ServiceAction
from .action import WaiterAction
from .base import ResourceMeta, ServiceResource
from .collection import CollectionFactory
from .model import ResourceModel
from .response import build_identifiers, ResourceHandler
from ..exceptions import ResourceLoadException
from ..docs import docstring
def _create_available_subresources_command(self, attrs, subresources):
    _subresources = [subresource.name for subresource in subresources]
    _subresources = sorted(_subresources)

    def get_available_subresources(factory_self):
        """
            Returns a list of all the available sub-resources for this
            Resource.

            :returns: A list containing the name of each sub-resource for this
                resource
            :rtype: list of str
            """
        return _subresources
    attrs['get_available_subresources'] = get_available_subresources