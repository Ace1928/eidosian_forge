import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
class ResourceFacade(function.Function):
    """A function for retrieving data in a parent provider template.

    A function for obtaining data from the facade resource from within the
    corresponding provider template.

    Takes the form::

        resource_facade: <attribute_type>

    where the valid attribute types are "metadata", "deletion_policy" and
    "update_policy".
    """
    _RESOURCE_ATTRIBUTES = METADATA, DELETION_POLICY, UPDATE_POLICY = ('metadata', 'deletion_policy', 'update_policy')

    def __init__(self, stack, fn_name, args):
        super(ResourceFacade, self).__init__(stack, fn_name, args)
        if self.args not in self._RESOURCE_ATTRIBUTES:
            fmt_data = {'fn_name': self.fn_name, 'allowed': ', '.join(self._RESOURCE_ATTRIBUTES)}
            raise ValueError(_('Incorrect arguments to "%(fn_name)s" should be one of: %(allowed)s') % fmt_data)

    def result(self):
        attr = function.resolve(self.args)
        if attr == self.METADATA:
            return self.stack.parent_resource.metadata_get()
        elif attr == self.UPDATE_POLICY:
            up = self.stack.parent_resource.t._update_policy or {}
            return function.resolve(up)
        elif attr == self.DELETION_POLICY:
            return self.stack.parent_resource.t.deletion_policy()