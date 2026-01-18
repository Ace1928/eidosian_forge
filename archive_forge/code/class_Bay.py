from heat.common.i18n import _
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
class Bay(none_resource.NoneResource):
    """A resource that creates a Magnum Bay.

    This resource has been deprecated in favor of OS::Magnum::Cluster.
    """
    deprecation_msg = _('Please use OS::Magnum::Cluster instead.')
    support_status = support.SupportStatus(status=support.HIDDEN, message=deprecation_msg, version='11.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, message=deprecation_msg, version='9.0.0', previous_status=support.SupportStatus(status=support.SUPPORTED, version='6.0.0')))