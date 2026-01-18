from . import errors, registry, urlutils
class InvalidBugTrackerURL(errors.BzrError):
    _fmt = 'The URL for bug tracker "%(abbreviation)s" doesn\'t contain {id}: %(url)s'

    def __init__(self, abbreviation, url):
        self.abbreviation = abbreviation
        self.url = url