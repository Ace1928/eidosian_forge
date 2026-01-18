from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient

        get available API versions
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_api_versions_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: V1APIGroupList
                 If the method is called asynchronously,
                 returns the request thread.
        