class InvalidHttpHeaderValue(Error):
    """Raised when an invalid HTTP header value is used.

  This issue arrises what a static handler uses http_headers. For example, the
  following would not be allowed:

    handlers:
    - url: /static
      static_dir: static
      http_headers:
        Some-Unicode: "☨"
  """