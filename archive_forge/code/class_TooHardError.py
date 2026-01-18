class TooHardError(RuntimeError):
    """max_work was exceeded.

    This is raised whenever the maximum number of candidate solutions
    to consider specified by the ``max_work`` parameter is exceeded.
    Assigning a finite number to max_work may have caused the operation
    to fail.

    """
    pass