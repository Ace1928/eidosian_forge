from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlRewrite(_messages.Message):
    """Defines the URL rewrite configuration for a given request.

  Fields:
    hostRewrite: Optional. Before forwarding the request to the selected
      origin, the request's host header is replaced with contents of
      `host_rewrite`. The host value must be between 1 and 255 characters.
    pathPrefixRewrite: Optional. Before forwarding the request to the selected
      origin, the matching portion of the request's path is replaced by
      `path_prefix_rewrite`. If specified, the path value must start with a
      `/` and be between 1 and 1024 characters long (inclusive).
      `path_prefix_rewrite` can only be used when all of a route's match_rules
      specify prefix_match or full_path_match. Only one of
      `path_prefix_rewrite` and path_template_rewrite can be specified.
    pathTemplateRewrite: Optional. Before forwarding the request to the
      selected origin, if the request matched a path_template_match, the
      matching portion of the request's path is replaced re-written using the
      pattern specified by `path_template_rewrite`. `path_template_rewrite`
      must be between 1 and 255 characters (inclusive), must start with a `/`,
      and must only use variables captured by the route's
      `path_template_match`. `path_template_rewrite` can only be used when all
      of a route's match_rules specify `path_template_match`. Only one of
      path_prefix_rewrite and `path_template_rewrite` can be specified.
  """
    hostRewrite = _messages.StringField(1)
    pathPrefixRewrite = _messages.StringField(2)
    pathTemplateRewrite = _messages.StringField(3)