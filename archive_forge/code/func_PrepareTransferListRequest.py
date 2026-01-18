from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def PrepareTransferListRequest(reference, location, page_size=None, page_token=None, data_source_ids=None):
    """Create and populate a list request."""
    request = dict(parent=FormatProjectIdentifierForTransfers(reference, location))
    if page_size is not None:
        request['pageSize'] = page_size
    if page_token is not None:
        request['pageToken'] = page_token
    if data_source_ids is not None:
        data_source_ids = data_source_ids.split(':')
        if data_source_ids[0] == 'dataSourceIds':
            data_source_ids = data_source_ids[1].split(',')
            request['dataSourceIds'] = data_source_ids
        else:
            raise bq_error.BigqueryError("Invalid filter flag values: '%s'. Expected format: '--filter=dataSourceIds:id1,id2'" % data_source_ids[0])
    return request