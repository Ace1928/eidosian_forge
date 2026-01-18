import logging
from oslo_vmware._i18n import _
class UseLinkedCloneConfigurationFault(VMwareDriverConfigurationException):
    msg_fmt = _('No default value for use_linked_clone found.')