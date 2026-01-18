class BzrBadParameter(InternalBzrError):
    _fmt = 'Bad parameter: %(param)r'

    def __init__(self, param):
        BzrError.__init__(self)
        self.param = param