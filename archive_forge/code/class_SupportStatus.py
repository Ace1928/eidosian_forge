from heat.common.i18n import _
class SupportStatus(object):

    def __init__(self, status=SUPPORTED, message=None, version=None, previous_status=None, substitute_class=None):
        """Use SupportStatus for current status of object.

        :param status: current status of object.
        :param version: version of OpenStack, from which current status is
                    valid. It may be None, but need to be defined for correct
                    doc generating.
        :param message: specific status message for object.
        :param substitute_class: assign substitute class.
        """
        self.status = status
        self.substitute_class = substitute_class
        self.message = message
        self.version = version
        self.previous_status = previous_status
        self.validate()

    def validate(self):
        if self.previous_status is not None and (not isinstance(self.previous_status, SupportStatus)):
            raise ValueError(_('previous_status must be SupportStatus instead of %s') % type(self.previous_status))
        if self.status not in SUPPORT_STATUSES:
            self.status = UNKNOWN
            self.message = _('Specified status is invalid, defaulting to %s') % UNKNOWN
            self.version = None
            self.previous_status = None

    def to_dict(self):
        return {'status': self.status, 'message': self.message, 'version': self.version, 'previous_status': self.previous_status.to_dict() if self.previous_status is not None else None}

    def is_substituted(self, substitute_class):
        if self.substitute_class is None:
            return False
        return substitute_class is self.substitute_class