import logging
from botocore import xform_name
class ResponseResource(object):
    """
    A resource response to create after performing an action.

    :type definition: dict
    :param definition: The JSON definition
    :type resource_defs: dict
    :param resource_defs: All resources defined in the service
    """

    def __init__(self, definition, resource_defs):
        self._definition = definition
        self._resource_defs = resource_defs
        self.type = definition.get('type')
        self.path = definition.get('path')

    @property
    def identifiers(self):
        """
        A list of resource identifiers.

        :type: list(:py:class:`Identifier`)
        """
        identifiers = []
        for item in self._definition.get('identifiers', []):
            identifiers.append(Parameter(**item))
        return identifiers

    @property
    def model(self):
        """
        Get the resource model for the response resource.

        :type: :py:class:`ResourceModel`
        """
        return ResourceModel(self.type, self._resource_defs[self.type], self._resource_defs)