from base64 import b64encode
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class OnAppResponse(JsonResponse):
    """
    OnApp response class
    """

    def success(self):
        """
        Determine if our request was successful.

        The meaning of this can be arbitrary; did we receive OK status? Did
        the node get created? Were we authenticated?

        :rtype: ``bool``
        :return: ``True`` or ``False``
        """
        return self.status in [httplib.OK, httplib.CREATED, httplib.NO_CONTENT]