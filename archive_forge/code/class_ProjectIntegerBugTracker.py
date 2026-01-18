from . import errors, registry, urlutils
class ProjectIntegerBugTracker(IntegerBugTracker):
    """A bug tracker that exists in one place only with per-project ids.

    If you have one of these trackers then register an instance passing in an
    abbreviated name for the bug tracker and a base URL. The bug ids are
    appended directly to the URL.
    """

    def __init__(self, abbreviated_bugtracker_name, base_url):
        self.abbreviation = abbreviated_bugtracker_name
        self._base_url = base_url

    def get(self, abbreviated_bugtracker_name, branch):
        """Returns the tracker if the abbreviation matches, otherwise ``None``.
        """
        if abbreviated_bugtracker_name != self.abbreviation:
            return None
        return self

    def check_bug_id(self, bug_id):
        try:
            project, bug_id = bug_id.rsplit('/', 1)
        except ValueError as exc:
            raise MalformedBugIdentifier(bug_id, 'Expected format: project/id') from exc
        try:
            int(bug_id)
        except ValueError as exc:
            raise MalformedBugIdentifier(bug_id, 'Bug id must be an integer') from exc

    def _get_bug_url(self, bug_id):
        project, bug_id = bug_id.rsplit('/', 1)
        'Return the URL for bug_id.'
        if '{id}' not in self._base_url:
            raise InvalidBugTrackerURL(self.abbreviation, self._base_url)
        if '{project}' not in self._base_url:
            raise InvalidBugTrackerURL(self.abbreviation, self._base_url)
        return self._base_url.replace('{project}', project).replace('{id}', str(bug_id))