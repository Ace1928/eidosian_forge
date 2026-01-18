from . import errors, registry, urlutils
class InvalidBugUrl(errors.BzrError):
    _fmt = 'Invalid bug URL: %(url)s'

    def __init__(self, url):
        self.url = url