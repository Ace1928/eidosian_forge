from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRedirectAction(_messages.Message):
    """Specifies settings for an HTTP redirect.

  Enums:
    RedirectResponseCodeValueValuesEnum: The HTTP Status code to use for this
      RedirectAction. Supported values are: - MOVED_PERMANENTLY_DEFAULT, which
      is the default value and corresponds to 301. - FOUND, which corresponds
      to 302. - SEE_OTHER which corresponds to 303. - TEMPORARY_REDIRECT,
      which corresponds to 307. In this case, the request method is retained.
      - PERMANENT_REDIRECT, which corresponds to 308. In this case, the
      request method is retained.

  Fields:
    hostRedirect: The host that is used in the redirect response instead of
      the one that was supplied in the request. The value must be from 1 to
      255 characters.
    httpsRedirect: If set to true, the URL scheme in the redirected request is
      set to HTTPS. If set to false, the URL scheme of the redirected request
      remains the same as that of the request. This must only be set for URL
      maps used in TargetHttpProxys. Setting this true for TargetHttpsProxy is
      not permitted. The default is set to false.
    pathRedirect: The path that is used in the redirect response instead of
      the one that was supplied in the request. pathRedirect cannot be
      supplied together with prefixRedirect. Supply one alone or neither. If
      neither is supplied, the path of the original request is used for the
      redirect. The value must be from 1 to 1024 characters.
    prefixRedirect: The prefix that replaces the prefixMatch specified in the
      HttpRouteRuleMatch, retaining the remaining portion of the URL before
      redirecting the request. prefixRedirect cannot be supplied together with
      pathRedirect. Supply one alone or neither. If neither is supplied, the
      path of the original request is used for the redirect. The value must be
      from 1 to 1024 characters.
    redirectResponseCode: The HTTP Status code to use for this RedirectAction.
      Supported values are: - MOVED_PERMANENTLY_DEFAULT, which is the default
      value and corresponds to 301. - FOUND, which corresponds to 302. -
      SEE_OTHER which corresponds to 303. - TEMPORARY_REDIRECT, which
      corresponds to 307. In this case, the request method is retained. -
      PERMANENT_REDIRECT, which corresponds to 308. In this case, the request
      method is retained.
    stripQuery: If set to true, any accompanying query portion of the original
      URL is removed before redirecting the request. If set to false, the
      query portion of the original URL is retained. The default is set to
      false.
  """

    class RedirectResponseCodeValueValuesEnum(_messages.Enum):
        """The HTTP Status code to use for this RedirectAction. Supported values
    are: - MOVED_PERMANENTLY_DEFAULT, which is the default value and
    corresponds to 301. - FOUND, which corresponds to 302. - SEE_OTHER which
    corresponds to 303. - TEMPORARY_REDIRECT, which corresponds to 307. In
    this case, the request method is retained. - PERMANENT_REDIRECT, which
    corresponds to 308. In this case, the request method is retained.

    Values:
      FOUND: Http Status Code 302 - Found.
      MOVED_PERMANENTLY_DEFAULT: Http Status Code 301 - Moved Permanently.
      PERMANENT_REDIRECT: Http Status Code 308 - Permanent Redirect
        maintaining HTTP method.
      SEE_OTHER: Http Status Code 303 - See Other.
      TEMPORARY_REDIRECT: Http Status Code 307 - Temporary Redirect
        maintaining HTTP method.
    """
        FOUND = 0
        MOVED_PERMANENTLY_DEFAULT = 1
        PERMANENT_REDIRECT = 2
        SEE_OTHER = 3
        TEMPORARY_REDIRECT = 4
    hostRedirect = _messages.StringField(1)
    httpsRedirect = _messages.BooleanField(2)
    pathRedirect = _messages.StringField(3)
    prefixRedirect = _messages.StringField(4)
    redirectResponseCode = _messages.EnumField('RedirectResponseCodeValueValuesEnum', 5)
    stripQuery = _messages.BooleanField(6)