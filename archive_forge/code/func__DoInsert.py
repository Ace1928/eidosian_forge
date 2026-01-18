from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import os
import sys
import textwrap
import time
from typing import Optional, TextIO
from absl import app
from absl import flags
import termcolor
import bq_flags
import bq_utils
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import stringutil
def _DoInsert(self, identifier: str, json_file: TextIO, skip_invalid_rows: Optional[bool]=None, ignore_unknown_values: Optional[bool]=None, template_suffix: Optional[int]=None, insert_id: Optional[str]=None) -> int:
    """Insert the contents of the file into a table."""
    client = bq_cached_client.Client.Get()
    reference = client.GetReference(identifier)
    bq_id_utils.typecheck(reference, (bq_id_utils.ApiClientHelper.TableReference,), 'Must provide a table identifier for insert.', is_usage_error=True)
    reference = dict(reference)
    batch = []

    def Flush():
        result = client.InsertTableRows(reference, batch, skip_invalid_rows=skip_invalid_rows, ignore_unknown_values=ignore_unknown_values, template_suffix=template_suffix)
        del batch[:]
        return (result, result.get('insertErrors', None))
    result = {}
    errors = None
    lineno = 1
    for line in json_file:
        try:
            unique_insert_id = None
            if insert_id is not None:
                unique_insert_id = insert_id + '_' + str(lineno)
            batch.append(bq_processor_utils.JsonToInsertEntry(unique_insert_id, line))
            lineno += 1
        except bq_error.BigqueryClientError as e:
            raise app.UsageError('Line %d: %s' % (lineno, str(e)))
        if FLAGS.max_rows_per_request and len(batch) == FLAGS.max_rows_per_request:
            result, errors = Flush()
        if errors:
            break
    if batch and (not errors):
        result, errors = Flush()
    if FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(result)
    elif FLAGS.format in [None, 'sparse', 'pretty']:
        if errors:
            for entry in result['insertErrors']:
                entry_errors = entry['errors']
                sys.stdout.write('record %d errors: ' % (entry['index'],))
                for error in entry_errors:
                    print('\t%s: %s' % (stringutil.ensure_str(error['reason']), stringutil.ensure_str(error.get('message'))))
    return 1 if errors else 0