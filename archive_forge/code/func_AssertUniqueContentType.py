from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def AssertUniqueContentType(self):
    """Makes sure that `self.http_headers` is consistent with `self.mime_type`.

    This method assumes that `self` is a static handler, either
    `self.static_dir` or `self.static_files`. You cannot specify `None`.

    Raises:
      appinfo_errors.ContentTypeSpecifiedMultipleTimes: If `self.http_headers`
          contains a `Content-Type` header, and `self.mime_type` is set. For
          example, the following configuration would be rejected::

              handlers:
              - url: /static
                static_dir: static
                mime_type: text/html
                http_headers:
                  content-type: text/html


        As this example shows, a configuration will be rejected when
        `http_headers` and `mime_type` specify a content type, even when they
        specify the same content type.
    """
    used_both_fields = self.mime_type and self.http_headers
    if not used_both_fields:
        return
    content_type = self.http_headers.Get('Content-Type')
    if content_type is not None:
        raise appinfo_errors.ContentTypeSpecifiedMultipleTimes('http_header specified a Content-Type header of %r in a handler that also specified a mime_type of %r.' % (content_type, self.mime_type))