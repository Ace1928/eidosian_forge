class BzrMoveFailedError(BzrError):
    _fmt = 'Could not move %(from_path)s%(operator)s %(to_path)s%(_has_extra)s%(extra)s'

    def __init__(self, from_path='', to_path='', extra=None):
        from breezy.osutils import splitpath
        BzrError.__init__(self)
        if extra:
            self.extra, self._has_extra = (extra, ': ')
        else:
            self.extra = self._has_extra = ''
        has_from = len(from_path) > 0
        has_to = len(to_path) > 0
        if has_from:
            self.from_path = splitpath(from_path)[-1]
        else:
            self.from_path = ''
        if has_to:
            self.to_path = splitpath(to_path)[-1]
        else:
            self.to_path = ''
        self.operator = ''
        if has_from and has_to:
            self.operator = ' =>'
        elif has_from:
            self.from_path = 'from ' + from_path
        elif has_to:
            self.operator = 'to'
        else:
            self.operator = 'file'