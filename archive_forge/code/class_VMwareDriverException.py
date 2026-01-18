import logging
from oslo_vmware._i18n import _
class VMwareDriverException(Exception):
    """Base oslo.vmware exception

    To correctly use this class, inherit from it and define
    a 'msg_fmt' property. That msg_fmt will get printf'd
    with the keyword arguments provided to the constructor.

    """
    msg_fmt = _('An unknown exception occurred.')

    def __str__(self):
        return self.description

    def __init__(self, message=None, details=None, **kwargs):
        if message is not None and isinstance(message, list):
            raise ValueError(_('exception message must not be a list'))
        if details is not None and (not isinstance(details, dict)):
            raise ValueError(_('details must be a dict'))
        self.kwargs = kwargs
        self.details = details
        self.cause = None
        if not message:
            try:
                message = self.msg_fmt % kwargs
            except Exception:
                LOG.exception('Exception in string format operation')
                for name, value in kwargs.items():
                    LOG.error('%(name)s: %(value)s', {'name': name, 'value': value})
                message = self.msg_fmt
        self.message = message
        super(VMwareDriverException, self).__init__(message)

    @property
    def msg(self):
        return self.message

    @property
    def description(self):
        descr = str(self.msg)
        if self.cause:
            descr += '\nCause: ' + str(self.cause)
        return descr