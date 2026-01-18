import jmespath
from botocore import xform_name
from .params import get_data_member
class RawHandler(object):
    """
    A raw action response handler. This passed through the response
    dictionary, optionally after performing a JMESPath search if one
    has been defined for the action.

    :type search_path: string
    :param search_path: JMESPath expression to search in the response
    :rtype: dict
    :return: Service response
    """

    def __init__(self, search_path):
        self.search_path = search_path

    def __call__(self, parent, params, response):
        """
        :type parent: ServiceResource
        :param parent: The resource instance to which this action is attached.
        :type params: dict
        :param params: Request parameters sent to the service.
        :type response: dict
        :param response: Low-level operation response.
        """
        if self.search_path and self.search_path != '$':
            response = jmespath.search(self.search_path, response)
        return response