from unittest import mock
from oslo_versionedobjects import exception
from oslo_versionedobjects import test
class TestWrapper(object):

    @exception.wrap_exception(notifier=notifier)
    def raise_exc(self, context, exc, admin_password):
        raise exc