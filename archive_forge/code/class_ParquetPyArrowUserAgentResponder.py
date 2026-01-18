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
class ParquetPyArrowUserAgentResponder(BaseUserAgentResponder):

    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        response_bytes = response_df.to_parquet(index=False, engine='pyarrow')
        self.write_back_bytes(response_bytes)