from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
def _ReadOnePage(self, start_row: Optional[int], max_rows: Optional[int], page_token: Optional[str]=None, selected_fields=None):
    kwds = dict(self.job_ref) if self.job_ref else {}
    kwds['maxResults'] = max_rows
    kwds['timeoutMs'] = 0
    if page_token:
        kwds['pageToken'] = page_token
    else:
        kwds['startIndex'] = start_row
    if not self._results['jobComplete']:
        raise bq_error.BigqueryError('Job %s is not done' % (self,))
    result_rows = self._results.get('rows', None)
    total_rows = self._results.get('totalRows', None)
    if total_rows is not None and result_rows is not None and (start_row is not None) and (len(result_rows) >= min(int(total_rows), start_row + max_rows)):
        page_token = self._results.get('pageToken', None)
        if len(result_rows) < int(total_rows) and page_token is None:
            raise bq_error.BigqueryError('Synchronous query %s did not return all rows, yet it did not return a page token' % (self,))
        schema = self._results.get('schema', None)
        rows = self._results.get('rows', [])
    else:
        data = self._apiclient.jobs().getQueryResults(**kwds).execute()
        if not data['jobComplete']:
            raise bq_error.BigqueryError('Job %s is not done' % (self,))
        page_token = data.get('pageToken', None)
        schema = data.get('schema', None)
        rows = data.get('rows', [])
    return (rows, page_token, schema)