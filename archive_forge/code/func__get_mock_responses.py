import random
import string
from tests.compat import unittest, mock
import boto
def _get_mock_responses(self, num, max_items):
    max_items = min(max_items, 100)
    cfid_groups = list(self._group_iter([self._get_random_id() for i in range(num)], max_items))
    cfg = dict(status='Completed', max_items=max_items, next_marker='')
    responses = []
    is_truncated = 'true'
    for i, group in enumerate(cfid_groups):
        next_marker = group[-1]
        if i + 1 == len(cfid_groups):
            is_truncated = 'false'
            next_marker = ''
        invals = ''
        cfg.update(dict(next_marker=next_marker, is_truncated=is_truncated))
        for cfid in group:
            cfg.update(dict(cfid=cfid))
            invals += INVAL_SUMMARY_TEMPLATE % cfg
        cfg.update(dict(inval_summaries=invals))
        mock_response = mock.Mock()
        mock_response.read.return_value = (RESPONSE_TEMPLATE % cfg).encode('utf-8')
        mock_response.status = 200
        responses.append(mock_response)
    return responses