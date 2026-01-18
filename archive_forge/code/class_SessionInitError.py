from oslo_limit._i18n import _
class SessionInitError(Exception):

    def __init__(self, reason):
        msg = _("Can't initialise OpenStackSDK session: %(reason)s.") % {'reason': reason}
        super(SessionInitError, self).__init__(msg)