import logging
from botocore import xform_name
class DefinitionWithParams(object):
    """
    An item which has parameters exposed via the ``params`` property.
    A request has an operation and parameters, while a waiter has
    a name, a low-level waiter name and parameters.

    :type definition: dict
    :param definition: The JSON definition
    """

    def __init__(self, definition):
        self._definition = definition

    @property
    def params(self):
        """
        Get a list of auto-filled parameters for this request.

        :type: list(:py:class:`Parameter`)
        """
        params = []
        for item in self._definition.get('params', []):
            params.append(Parameter(**item))
        return params