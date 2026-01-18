from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class CombinationAlarm(none_resource.NoneResource):
    """A resource that implements combination of Aodh alarms.

    This resource is now deleted from Aodh, so will directly inherit from
    NoneResource (placeholder resource). For old resources (which not a
    placeholder resource), still can be deleted through client. Any newly
    created resources will be considered as placeholder resources like none
    resource. We will schedule to delete it from heat resources list.
    """
    default_client_name = 'aodh'
    entity = 'alarm'
    support_status = support.SupportStatus(status=support.HIDDEN, message=_('OS::Aodh::CombinationAlarm is deprecated and has been removed from Aodh, use OS::Aodh::CompositeAlarm instead.'), version='9.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='7.0.0', previous_status=support.SupportStatus(version='2014.1')))