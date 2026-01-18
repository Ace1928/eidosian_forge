class BadBundle(BzrError):
    _fmt = 'Bad bzr revision-bundle: %(text)r'

    def __init__(self, text):
        BzrError.__init__(self)
        self.text = text