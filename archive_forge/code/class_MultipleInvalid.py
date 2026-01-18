class MultipleInvalid(_TargetInvalid):
    """
    The *target* has failed to implement the *interface* in
    multiple ways.

    The failures are described by *exceptions*, a collection of
    other `Invalid` instances.

    .. versionadded:: 5.0
    """
    _NOT_GIVEN_CATCH = ()

    def __init__(self, interface, target, exceptions):
        super().__init__(interface, target, tuple(exceptions))

    @property
    def exceptions(self):
        return self.args[2]

    @property
    def _str_details(self):
        return '\n    ' + '\n    '.join((x._str_details.strip() if isinstance(x, _TargetInvalid) else str(x) for x in self.exceptions))
    _str_conjunction = ':'
    _str_trailer = ''