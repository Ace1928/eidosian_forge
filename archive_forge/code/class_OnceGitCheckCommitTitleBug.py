import os
import re
import subprocess  # nosec
from hacking import core
class OnceGitCheckCommitTitleBug(GitCheck):
    """Check git commit messages for bugs.

    OpenStack HACKING recommends not referencing a bug or blueprint in first
    line. It should provide an accurate description of the change
    S364
    """
    name = 'GitCheckCommitTitleBug'
    GIT_REGEX = re.compile('(I[0-9a-f]{8,40})|([Bb]ug|[Ll][Pp])[\\s\\#:]*(\\d+)|([Bb]lue[Pp]rint|[Bb][Pp])[\\s\\#:]*([A-Za-z0-9\\\\-]+)')

    def run_once(self):
        title = self._get_commit_title()
        if title and self.GIT_REGEX.search(title) is not None and (len(title.split()) <= 3):
            return (1, 0, "S364: git commit title ('%s') should provide an accurate description of the change, not just a reference to a bug or blueprint" % title.strip(), self.name)