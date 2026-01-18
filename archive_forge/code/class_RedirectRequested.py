class RedirectRequested(TransportError):
    _fmt = '%(source)s is%(permanently)s redirected to %(target)s'

    def __init__(self, source, target, is_permanent=False):
        self.source = source
        self.target = target
        if is_permanent:
            self.permanently = ' permanently'
        else:
            self.permanently = ''
        TransportError.__init__(self)