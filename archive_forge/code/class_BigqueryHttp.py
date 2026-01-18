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
class BigqueryHttp(http_request.HttpRequest):
    """Converts errors into Bigquery errors."""

    def __init__(self, bigquery_model: BigqueryModel, *args, **kwds):
        super().__init__(*args, **kwds)
        logging.info('URL being requested from BQ client: %s %s', kwds['method'], args[2])
        self._model = bigquery_model

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

    @staticmethod
    def RaiseErrorFromHttpError(e):
        """Raises a BigQueryError given an HttpError."""
        return bq_client_utils.RaiseErrorFromHttpError(e)

    @staticmethod
    def RaiseErrorFromNonHttpError(e):
        """Raises a BigQueryError given a non-HttpError."""
        return bq_client_utils.RaiseErrorFromNonHttpError(e)

    def execute(self, **kwds):
        try:
            return super().execute(**kwds)
        except googleapiclient.errors.HttpError as e:
            self._model._log_response(e.resp, e.content)
            bq_client_utils.RaiseErrorFromHttpError(e)
        except (httplib2.HttpLib2Error, IOError) as e:
            bq_client_utils.RaiseErrorFromNonHttpError(e)