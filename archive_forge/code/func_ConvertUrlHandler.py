from __future__ import absolute_import
import re
def ConvertUrlHandler(handler):
    """Rejiggers the structure of the url handler based on its type.

  An extra level of message nesting occurs here, based on the handler type.
  Fields common to all handler types occur at the top-level, while
  handler-specific fields will go into a submessage based on handler type.

  For example, a static files handler is transformed as follows:
  Input {
    "urlRegex": "foo/bar.html",
    "path": "static_files/foo/bar.html"
  }
  Output {
    "urlRegex": "foo/bar.html",
    "staticFiles": {
      "path": "static_files/foo/bar.html"
    }
  }

  Args:
    handler: Result of converting handler according to schema.

  Returns:
    Handler which has moved fields specific to the handler's type to a
    submessage.
  """

    def AppendRegexToPath(path, regex):
        """Equivalent to os.path.join(), except uses forward slashes always."""
        return path.rstrip('/') + '/' + regex
    handler_type = _GetHandlerType(handler)
    if handler_type == 'staticDirectory':
        try:
            compiled = re.compile(handler['urlRegex'])
        except re.error:
            pass
        else:
            if compiled.groups:
                raise ValueError('Groups are not allowed in URLs for static directory handlers: ' + handler['urlRegex'])
        tmp = {'path': AppendRegexToPath(handler['staticDir'], '\\1'), 'uploadPathRegex': AppendRegexToPath(handler['staticDir'], '.*'), 'urlRegex': AppendRegexToPath(handler['urlRegex'], '(.*)')}
        del handler['staticDir']
        handler.update(tmp)
        handler_type = 'staticFiles'
    new_handler = {}
    new_handler[handler_type] = {}
    for field in _HANDLER_FIELDS[handler_type]:
        if field in handler:
            new_handler[handler_type][field] = handler[field]
    for common_field in _COMMON_HANDLER_FIELDS:
        if common_field in handler:
            new_handler[common_field] = handler[common_field]
    return new_handler