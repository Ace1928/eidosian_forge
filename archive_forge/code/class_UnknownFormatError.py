class UnknownFormatError(BzrError):
    _fmt = 'Unknown %(kind)s format: %(format)r'

    def __init__(self, format, kind='branch'):
        self.kind = kind
        self.format = format