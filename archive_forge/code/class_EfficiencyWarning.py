class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """