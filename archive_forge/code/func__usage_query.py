import oslo_utils
from novaclient import api_versions
from novaclient import base
def _usage_query(self, start, end, marker=None, limit=None, detailed=None):
    query = '?start=%s&end=%s' % (start.isoformat(), end.isoformat())
    if limit:
        query = '%s&limit=%s' % (query, int(limit))
    if marker:
        query = '%s&marker=%s' % (query, marker)
    if detailed is not None:
        query = '%s&detailed=%s' % (query, int(bool(detailed)))
    return query