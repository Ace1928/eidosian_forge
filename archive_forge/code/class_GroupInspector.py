from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
class GroupInspector(object):
    """A class for returning data about a scaling group.

    All data is fetched over RPC, and the group's stack is never loaded into
    memory locally. Data is cached so it will be fetched only once. To
    refresh the data, create a new GroupInspector.
    """

    def __init__(self, context, rpc_client, group_identity):
        """Initialise with a context, rpc_client, and stack identifier."""
        self._context = context
        self._rpc_client = rpc_client
        self._identity = group_identity
        self._member_data = None
        self._template_data = None

    @classmethod
    def from_parent_resource(cls, parent_resource):
        """Create a GroupInspector from a parent resource.

        This is a convenience method to instantiate a GroupInspector from a
        Heat StackResource object.
        """
        return cls(parent_resource.context, parent_resource.rpc_client(), parent_resource.nested_identifier())

    def _get_member_data(self):
        if self._identity is None:
            return []
        if self._member_data is None:
            rsrcs = self._rpc_client.list_stack_resources(self._context, dict(self._identity))

            def sort_key(r):
                return (r[rpc_api.RES_STATUS] != status.ResourceStatus.FAILED, r[rpc_api.RES_CREATION_TIME], r[rpc_api.RES_NAME])
            self._member_data = sorted(rsrcs, key=sort_key)
        return self._member_data

    def _members(self, include_failed):
        return (r for r in self._get_member_data() if include_failed or r[rpc_api.RES_STATUS] != status.ResourceStatus.FAILED)

    def size(self, include_failed):
        """Return the size of the group.

        If include_failed is False, only members not in a FAILED state will
        be counted.
        """
        return sum((1 for m in self._members(include_failed)))

    def member_names(self, include_failed):
        """Return an iterator over the names of the group members

        If include_failed is False, only members not in a FAILED state will
        be included.
        """
        return (m[rpc_api.RES_NAME] for m in self._members(include_failed))

    def _get_template_data(self):
        if self._identity is None:
            return None
        if self._template_data is None:
            self._template_data = self._rpc_client.get_template(self._context, self._identity)
        return self._template_data

    def template(self):
        """Return a Template object representing the group's current template.

        Note that this does *not* include any environment data.
        """
        data = self._get_template_data()
        if data is None:
            return None
        return template.Template(data)