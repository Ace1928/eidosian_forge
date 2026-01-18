import abc
import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
from requests import Response
from cinderclient.apiclient import exceptions
from cinderclient import utils
class ListWithMeta(list, RequestIdMixin):

    def __init__(self, values, resp):
        super(ListWithMeta, self).__init__(values)
        self.setup()
        self.append_request_ids(resp)