import abc
import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
from requests import Response
from cinderclient.apiclient import exceptions
from cinderclient import utils
class TupleWithMeta(tuple, RequestIdMixin):

    def __new__(cls, values, resp):
        return super(TupleWithMeta, cls).__new__(cls, values)

    def __init__(self, values, resp):
        self.setup()
        self.append_request_ids(resp)