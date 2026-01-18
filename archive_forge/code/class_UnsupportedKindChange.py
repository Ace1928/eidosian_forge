class UnsupportedKindChange(BzrError):
    _fmt = 'Kind change from %(from_kind)s to %(to_kind)s for %(path)s not supported by format %(format)r'

    def __init__(self, path, from_kind, to_kind, format):
        self.path = path
        self.from_kind = from_kind
        self.to_kind = to_kind
        self.format = format