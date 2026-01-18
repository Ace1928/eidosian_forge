class DoesNotImplement(_TargetInvalid):
    """
    DoesNotImplement(interface[, target])

    The *target* (optional) does not implement the *interface*.

    .. versionchanged:: 5.0.0
       Add the *target* argument and attribute, and change the resulting
       string value of this object accordingly.
    """
    _str_details = 'Does not declaratively implement the interface'