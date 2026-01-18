class ContentTypeSpecifiedMultipleTimes(Error):
    """Raised when mime_type and http_headers specify a mime type.

  N.B. This will be raised even when both fields specify the same content type.
  E.g. the following configuration (snippet) will be rejected:

    handlers:
    - url: /static
      static_dir: static
      mime_type: text/html
      http_headers:
        content-type: text/html

  This only applies to static handlers i.e. a handler that specifies static_dir
  or static_files.
  """