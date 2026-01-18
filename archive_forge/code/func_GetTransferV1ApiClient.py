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
def GetTransferV1ApiClient(self, transferserver_address: Optional[str]=None) -> discovery.Resource:
    """Return the apiclient that supports Transfer v1 operation."""
    logging.info('GetTransferV1ApiClient transferserver_address: %s', transferserver_address)
    if self._op_transfer_client:
        logging.info('Using the cached Transfer API client')
    else:
        path = transferserver_address or bq_api_utils.get_tpc_root_url_from_flags(service=Service.DTS, inputted_flags=bq_flags, local_params=self)
        discovery_url = bq_api_utils.get_discovery_url_from_root_url(path, api_version='v1')
        self._op_transfer_client = self.BuildApiClient(discovery_url=discovery_url, service=Service.DTS)
    return self._op_transfer_client