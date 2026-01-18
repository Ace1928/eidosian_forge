class OsloDBDeprecationWarning(DeprecationWarning):
    """Issued per usage of a deprecated API.

    This subclasses DeprecationWarning so that it can be filtered as a distinct
    category.

    .. seealso::

        https://docs.python.org/2/library/warnings.html

    """