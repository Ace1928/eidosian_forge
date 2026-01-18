from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidConfigurationOption(NeutronException):
    """An error due to an invalid configuration option value.

    :param opt_name: The name of the configuration option that has an invalid
        value.
    :param opt_value: The value that's invalid for the configuration option.
    """
    message = _('An invalid value was provided for %(opt_name)s: %(opt_value)s.')