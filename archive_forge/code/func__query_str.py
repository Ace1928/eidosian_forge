from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
def _query_str(self, data):
    """Return the query fragment of a signed URI.

        This can be used, for example, for alarming.
        """
    paths = jsonutils.loads(data[self.PATHS_ATTR])
    methods = jsonutils.loads(data[self.METHODS_ATTR])
    query = {'signature': data[self.SIGNATURE], 'expires': data[self.EXPIRES], 'paths': ','.join(paths), 'methods': ','.join(methods), 'project_id': data[self.PROJECT], 'queue_name': self.properties[self.QUEUE]}
    return parse.urlencode(query)