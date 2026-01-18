import h2.errors
class RFC1122Error(H2Error):
    """
    Emitted when users attempt to do something that is literally allowed by the
    relevant RFC, but is sufficiently ill-defined that it's unwise to allow
    users to actually do it.

    While there is some disagreement about whether or not we should be liberal
    in what accept, it is a truth universally acknowledged that we should be
    conservative in what emit.

    .. versionadded:: 2.4.0
    """
    pass