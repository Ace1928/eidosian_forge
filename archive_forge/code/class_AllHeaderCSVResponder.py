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
class AllHeaderCSVResponder(http.server.BaseHTTPRequestHandler):
    """
    Send all request headers back for checking round trip
    """

    def do_GET(self):
        response_df = pd.DataFrame(self.headers.items())
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv')
        self.end_headers()
        response_bytes = response_df.to_csv(index=False).encode('utf-8')
        self.wfile.write(response_bytes)