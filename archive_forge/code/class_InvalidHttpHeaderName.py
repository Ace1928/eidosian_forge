class InvalidHttpHeaderName(Error):
    """Raised when an invalid HTTP header name is used.

  This issue arrises what a static handler uses http_headers. For example, the
  following would not be allowed:

    handlers:
    - url: /static
      static_dir: static
      http_headers:
        D@nger: Will Robinson
  """