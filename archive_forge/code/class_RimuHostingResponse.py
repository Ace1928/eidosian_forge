from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
class RimuHostingResponse(JsonResponse):
    """
    Response Class for RimuHosting driver
    """

    def success(self):
        if self.status == 403:
            raise InvalidCredsError()
        return True

    def parse_body(self):
        try:
            js = super().parse_body()
            keys = list(js.keys())
            if js[keys[0]]['response_type'] == 'ERROR':
                raise RimuHostingException(js[keys[0]]['human_readable_message'])
            return js[keys[0]]
        except KeyError:
            raise RimuHostingException('Could not parse body: %s' % self.body)