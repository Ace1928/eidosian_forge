from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
def AddOrderedResponses(self, url, response_funcs):
    """Calls the provided function when the provided URL is requested.

      The provided functions should accept a request object and return a
      response object. Each response will be called once.

      Args:
        url: The URL to trigger on.
        response_funcs: A list of response functions.
      """
    self.ordered_responses[url] = response_funcs