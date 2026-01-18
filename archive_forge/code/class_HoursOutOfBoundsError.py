class HoursOutOfBoundsError(RangeCheckError):
    """Raise when parsed hours are greater than 24."""