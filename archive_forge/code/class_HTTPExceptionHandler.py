from paste.wsgilib import catch_errors_app
from paste.response import has_header, header_value, replace_header
from paste.request import resolve_relative_url
from paste.util.quoting import strip_html, html_quote, no_quote, comment_quote
class HTTPExceptionHandler(object):
    """
    catches exceptions and turns them into proper HTTP responses

    This middleware catches any exceptions (which are subclasses of
    ``HTTPException``) and turns them into proper HTTP responses.
    Note if the headers have already been sent, the stack trace is
    always maintained as this indicates a programming error.

    Note that you must raise the exception before returning the
    app_iter, and you cannot use this with generator apps that don't
    raise an exception until after their app_iter is iterated over.
    """

    def __init__(self, application, warning_level=None):
        assert not warning_level or (warning_level > 99 and warning_level < 600)
        if warning_level is not None:
            import warnings
            warnings.warn('The warning_level parameter is not used or supported', DeprecationWarning, 2)
        self.warning_level = warning_level or 500
        self.application = application

    def __call__(self, environ, start_response):
        environ['paste.httpexceptions'] = self
        environ.setdefault('paste.expected_exceptions', []).append(HTTPException)
        try:
            return self.application(environ, start_response)
        except HTTPException as exc:
            return exc(environ, start_response)