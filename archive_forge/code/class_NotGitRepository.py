import binascii
class NotGitRepository(Exception):
    """Indicates that no Git repository was found."""

    def __init__(self, *args, **kwargs) -> None:
        Exception.__init__(self, *args, **kwargs)