from openstack.cloud import _utils
from openstack import exceptions
from openstack.orchestration.util import event_utils
from openstack.orchestration.v1._proxy import Proxy
Get exactly one stack.

        :param name_or_id: Name or ID of the desired stack.
        :param filters: a dict containing additional filters to use. e.g.
                {'stack_status': 'CREATE_COMPLETE'}
        :param resolve_outputs: If True, then outputs for this
                stack will be resolved

        :returns: a :class:`openstack.orchestration.v1.stack.Stack`
            containing the stack description
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call or if multiple matches are
            found.
        