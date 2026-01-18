from mlflow import __version__
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
class DefaultRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides default request headers for outgoing request.
    """

    def in_context(self):
        return True

    def request_headers(self):
        return dict(**_DEFAULT_HEADERS)