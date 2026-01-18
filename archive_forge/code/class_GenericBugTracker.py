from . import errors, registry, urlutils
class GenericBugTracker(URLParametrizedBugTracker):
    """Generic bug tracker specified by an URL template."""

    def __init__(self):
        super().__init__('bugtracker', None)

    def get(self, abbreviation, branch):
        self._abbreviation = abbreviation
        return super().get(abbreviation, branch)

    def _get_bug_url(self, bug_id):
        """Given a validated bug_id, return the bug's web page's URL."""
        if '{id}' not in self._base_url:
            raise InvalidBugTrackerURL(self._abbreviation, self._base_url)
        return self._base_url.replace('{id}', str(bug_id))