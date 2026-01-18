from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
from typing import Optional
from absl import flags
import googleapiclient
from googleapiclient import http as http_request
from googleapiclient import model
import httplib2
import bq_utils
from clients import utils as bq_client_utils
@staticmethod
def Factory(bigquery_model: BigqueryModel, use_google_auth: bool):
    """Returns a function that creates a BigqueryHttp with the given model."""

    def _Construct(*args, **kwds):
        if use_google_auth:
            user_agent = bq_utils.GetUserAgent()
            if 'headers' not in kwds:
                kwds['headers'] = {}
            elif 'User-Agent' in kwds['headers'] and user_agent not in kwds['headers']['User-Agent']:
                user_agent = ' '.join([user_agent, kwds['headers']['User-Agent']])
            kwds['headers']['User-Agent'] = user_agent
        captured_model = bigquery_model
        return BigqueryHttp(captured_model, *args, **kwds)
    return _Construct