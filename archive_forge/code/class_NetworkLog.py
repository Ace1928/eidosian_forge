import collections
import copy
from unittest import mock
import uuid
class NetworkLog(FakeLogging):
    """Fake one or more network log"""

    def __init__(self):
        super(NetworkLog, self).__init__()
        self.ordered = collections.OrderedDict((('id', 'log-id-' + uuid.uuid4().hex), ('description', 'my-desc-' + uuid.uuid4().hex), ('enabled', False), ('name', 'my-log-' + uuid.uuid4().hex), ('target_id', None), ('project_id', 'project-id-' + uuid.uuid4().hex), ('resource_id', None), ('resource_type', 'security_group'), ('event', 'all')))