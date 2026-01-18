import gzip
import http.server
from io import BytesIO
import multiprocessing
import socket
import time
import urllib.error
import pytest
from pandas.compat import is_ci_environment
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
class BaseUserAgentResponder(http.server.BaseHTTPRequestHandler):
    """
    Base class for setting up a server that can be set up to respond
    with a particular file format with accompanying content-type headers.
    The interfaces on the different io methods are different enough
    that this seemed logical to do.
    """

    def start_processing_headers(self):
        """
        shared logic at the start of a GET request
        """
        self.send_response(200)
        self.requested_from_user_agent = self.headers['User-Agent']
        response_df = pd.DataFrame({'header': [self.requested_from_user_agent]})
        return response_df

    def gzip_bytes(self, response_bytes):
        """
        some web servers will send back gzipped files to save bandwidth
        """
        with BytesIO() as bio:
            with gzip.GzipFile(fileobj=bio, mode='w') as zipper:
                zipper.write(response_bytes)
            response_bytes = bio.getvalue()
        return response_bytes

    def write_back_bytes(self, response_bytes):
        """
        shared logic at the end of a GET request
        """
        self.wfile.write(response_bytes)