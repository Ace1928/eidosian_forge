from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
def AddOrderedResponse(self, url, response_func):
    """Calls the provided function when the provided URL is requested.

      The provided functions should accept a request object and return a
      response object.  This response will be added after previously given
      responses if they exist.

      Args:
        url: The URL to trigger on.
        response_func: The function to call when the url is requested.
      """
    if url not in self.ordered_responses:
        self.ordered_responses[url] = []
    self.ordered_responses[url].append(response_func)