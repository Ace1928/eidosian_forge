import binascii
class MissingCommitError(Exception):
    """Indicates that a commit was not found in the repository."""

    def __init__(self, sha, *args, **kwargs) -> None:
        self.sha = sha
        Exception.__init__(self, '%s is not in the revision store' % sha)