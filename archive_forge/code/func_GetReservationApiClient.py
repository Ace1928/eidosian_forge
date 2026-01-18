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
def GetReservationApiClient(self, reservationserver_address: Optional[str]=None) -> discovery.Resource:
    """Return the apiclient that supports reservation operations."""
    if self._op_reservation_client:
        logging.info('Using the cached Reservations API client')
    else:
        path = reservationserver_address or bq_api_utils.get_tpc_root_url_from_flags(service=Service.RESERVATIONS, inputted_flags=bq_flags, local_params=self)
        reservation_version = 'v1'
        discovery_url = bq_api_utils.get_discovery_url_from_root_url(path, api_version=reservation_version)
        self._op_reservation_client = self.BuildApiClient(discovery_url=discovery_url, service=Service.RESERVATIONS)
    return self._op_reservation_client