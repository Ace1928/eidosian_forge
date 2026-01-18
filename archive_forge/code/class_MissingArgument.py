from wsme.utils import _
class MissingArgument(ClientSideError):

    def __init__(self, argname, msg=''):
        self.argname = argname
        super(MissingArgument, self).__init__(msg)

    @property
    def faultstring(self):
        return _('Missing argument: "%s"%s') % (self.argname, self.msg and ': ' + self.msg or '')