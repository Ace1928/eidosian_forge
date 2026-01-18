from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from http import client as http_client_lib
import json
import logging
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Union
import urllib
from absl import flags
import googleapiclient
from googleapiclient import discovery
import httplib2
import bq_flags
import bq_utils
from clients import bigquery_http
from clients import utils as bq_client_utils
from discovery_documents import discovery_document_cache
from discovery_documents import discovery_document_loader
from utils import bq_api_utils
from utils import bq_error
def BuildApiClient(self, service: Service, discovery_url: Optional[str]=None) -> discovery.Resource:
    """Build and return BigQuery Dynamic client from discovery document."""
    logging.info('BuildApiClient discovery_url: %s', discovery_url)
    http_client = self.GetHttp()
    http = self.GetAuthorizedHttp(self.credentials, http_client, is_for_discovery=True)
    bigquery_model = bigquery_http.BigqueryModel(trace=self.trace, quota_project_id=bq_utils.GetEffectiveQuotaProjectIDForHTTPHeader(self.quota_project_id, self.use_google_auth, self.credentials))
    bq_request_builder = bigquery_http.BigqueryHttp.Factory(bigquery_model, self.use_google_auth)
    discovery_document = None
    if self.discovery_document != _DEFAULT:
        discovery_document = self.discovery_document
        logging.info('Skipping local discovery document load since discovery_document has a value: %s', discovery_document)
    elif discovery_url is not None:
        logging.info('Skipping local discovery document load since discovery_url has a value')
    elif self.api not in discovery_document_loader.SUPPORTED_BIGQUERY_APIS or self.api_version != 'v2':
        logging.info('Loading discovery doc from the server since this is not v2 (%s) and the API endpoint (%s) is not one of (%s).', self.api_version, self.api, ', '.join(discovery_document_loader.SUPPORTED_BIGQUERY_APIS))
    else:
        try:
            discovery_document = discovery_document_loader.load_local_discovery_doc(discovery_document_loader.DISCOVERY_NEXT_BIGQUERY)
        except FileNotFoundError as e:
            logging.warning('Failed to load discovery doc from local files: %s', e)
    if discovery_document is not None:
        logging.info('Discovery doc is already loaded')
    else:
        max_retries = 3
        iterations = 0
        headers = {'X-ESF-Use-Cloud-UberMint-If-Enabled': '1'} if hasattr(self, 'use_uber_mint') and self.use_uber_mint else None
        while iterations < max_retries and discovery_document is None:
            if iterations > 0:
                time.sleep(2 ** iterations)
            iterations += 1
            try:
                if discovery_url is None:
                    discovery_url = self.GetDiscoveryUrl(service=service, api_version=self.api_version)
                logging.info('Requesting discovery document from %s', discovery_url)
                if headers:
                    response_metadata, discovery_document = http.request(discovery_url, headers=headers)
                else:
                    response_metadata, discovery_document = http.request(discovery_url)
                discovery_document = discovery_document.decode('utf-8')
                if int(response_metadata.get('status')) >= 400:
                    msg = 'Got %s response from discovery url: %s' % (response_metadata.get('status'), discovery_url)
                    logging.error('%s:\n%s', msg, discovery_document)
                    raise bq_error.BigqueryCommunicationError(msg)
            except (httplib2.HttpLib2Error, googleapiclient.errors.HttpError, http_client_lib.HTTPException) as e:
                if hasattr(e, 'content'):
                    if iterations == max_retries:
                        content = ''
                        if hasattr(e, 'content'):
                            content = e.content
                        raise bq_error.BigqueryCommunicationError('Cannot contact server. Please try again.\nError: %r\nContent: %s' % (e, content))
                elif iterations == max_retries:
                    raise bq_error.BigqueryCommunicationError('Cannot contact server. Please try again.\nTraceback: %s' % (traceback.format_exc(),))
            except IOError as e:
                if iterations == max_retries:
                    raise bq_error.BigqueryCommunicationError('Cannot contact server. Please try again.\nError: %r' % (e,))
            except googleapiclient.errors.UnknownApiNameOrVersion as e:
                raise bq_error.BigqueryCommunicationError('Invalid API name or version: %s' % (str(e),))
    discovery_document_to_build_client = self.OverrideEndpoint(discovery_document=discovery_document, service=service)
    built_client = None
    try:
        built_client = discovery.build_from_document(discovery_document_to_build_client, http=http, model=bigquery_model, requestBuilder=bq_request_builder)
    except Exception:
        logging.error('Error building from discovery document: %s', discovery_document)
        raise
    return built_client