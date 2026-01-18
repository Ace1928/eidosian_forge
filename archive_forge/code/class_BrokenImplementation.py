class BrokenImplementation(_TargetInvalid):
    """
    BrokenImplementation(interface, name[, target])

    The *target* (optional) is missing the attribute *name*.

    .. versionchanged:: 5.0.0
       Add the *target* argument and attribute, and change the resulting
       string value of this object accordingly.

       The *name* can either be a simple string or a ``Attribute`` object.
    """
    _IX_NAME = _TargetInvalid._IX_INTERFACE + 1
    _IX_TARGET = _IX_NAME + 1

    @property
    def name(self):
        return self.args[1]

    @property
    def _str_details(self):
        return 'The %s attribute was not provided' % (repr(self.name) if isinstance(self.name, str) else self.name)