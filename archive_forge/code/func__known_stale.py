import os, subprocess, json
def _known_stale(self):
    """
        The commit is known to be from a file (and therefore stale) if a
        SHA is supplied by git archive and doesn't match the parsed commit.
        """
    if self._output_from_file() is None:
        commit = None
    else:
        commit = self.commit
    known_stale = self.archive_commit is not None and (not self.archive_commit.startswith('$Format')) and (self.archive_commit != commit)
    if known_stale:
        self._commit_count = None
    return known_stale