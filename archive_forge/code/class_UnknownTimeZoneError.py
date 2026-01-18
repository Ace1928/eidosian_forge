class UnknownTimeZoneError(KeyError, Error):
    """Exception raised when pytz is passed an unknown timezone.

    >>> isinstance(UnknownTimeZoneError(), LookupError)
    True

    This class is actually a subclass of KeyError to provide backwards
    compatibility with code relying on the undocumented behavior of earlier
    pytz releases.

    >>> isinstance(UnknownTimeZoneError(), KeyError)
    True

    And also a subclass of pytz.exceptions.Error, as are other pytz
    exceptions.

    >>> isinstance(UnknownTimeZoneError(), Error)
    True

    """
    pass