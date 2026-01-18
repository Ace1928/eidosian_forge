from pprint import pformat
from six import iteritems
import re
@_continue.setter
def _continue(self, _continue):
    """
        Sets the _continue of this V1ListMeta.
        continue may be set if the user set a limit on the number of items
        returned, and indicates that the server has more data available. The
        value is opaque and may be used to issue another request to the endpoint
        that served this list to retrieve the next set of available objects.
        Continuing a consistent list may not be possible if the server
        configuration has changed or more than a few minutes have passed. The
        resourceVersion field returned when using this continue value will be
        identical to the value in the first response, unless you have received
        this token from an error message.

        :param _continue: The _continue of this V1ListMeta.
        :type: str
        """
    self.__continue = _continue