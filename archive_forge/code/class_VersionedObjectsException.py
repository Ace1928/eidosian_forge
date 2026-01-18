import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class VersionedObjectsException(Exception):
    """Base VersionedObjects Exception

    To correctly use this class, inherit from it and define
    a 'msg_fmt' property. That msg_fmt will get printf'd
    with the keyword arguments provided to the constructor.

    """
    msg_fmt = _('An unknown exception occurred.')
    code = 500
    headers = {}
    safe = False

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if 'code' not in self.kwargs:
            try:
                self.kwargs['code'] = self.code
            except AttributeError:
                pass
        if not message:
            try:
                message = self.msg_fmt % kwargs
            except Exception:
                LOG.exception('Exception in string format operation')
                for name, value in kwargs.items():
                    LOG.error('%s: %s' % (name, value))
                if CONF.oslo_versionedobjects.fatal_exception_format_errors:
                    raise
                else:
                    message = self.msg_fmt
        super(VersionedObjectsException, self).__init__(message)

    def format_message(self):
        return self.args[0]