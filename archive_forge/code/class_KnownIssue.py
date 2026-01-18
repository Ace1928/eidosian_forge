class KnownIssue(Exception):
    """
    Raised in case of an known problem. Mostly because of cpython bugs.
    Executing.node gets set to None in this case.
    """
    pass