from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def JsonToInsertEntry(insert_id: Optional[str], json_string: str) -> InsertEntry:
    """Parses a JSON encoded record and returns an InsertEntry.

  Arguments:
    insert_id: Id for the insert, can be None.
    json_string: The JSON encoded data to be converted.

  Returns:
    InsertEntry object for adding to a table.
  """
    try:
        row = json.loads(json_string)
        if not isinstance(row, dict):
            raise bq_error.BigqueryClientError('Value is not a JSON object')
        return InsertEntry(insert_id, row)
    except ValueError as e:
        raise bq_error.BigqueryClientError('Could not parse object: %s' % (str(e),))