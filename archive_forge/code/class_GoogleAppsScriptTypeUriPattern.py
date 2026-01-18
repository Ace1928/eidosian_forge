from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeUriPattern(_messages.Message):
    """The configuration for each URL pattern that triggers a link preview.

  Fields:
    hostPattern: Required for each URL pattern to preview. The domain of the
      URL pattern. The add-on previews links that contain this domain in the
      URL. To preview links for a specific subdomain, like
      `subdomain.example.com`, include the subdomain. To preview links for the
      entire domain, specify a wildcard character with an asterisk (`*`) as
      the subdomain. For example, `*.example.com` matches
      `subdomain.example.com` and `another.subdomain.example.com`.
    pathPrefix: Optional. The path that appends the domain of the
      `hostPattern`. For example, if the URL host pattern is
      `support.example.com`, to match URLs for cases hosted at
      `support.example.com/cases/`, enter `cases`. To match all URLs in the
      host pattern domain, leave `pathPrefix` empty.
  """
    hostPattern = _messages.StringField(1)
    pathPrefix = _messages.StringField(2)