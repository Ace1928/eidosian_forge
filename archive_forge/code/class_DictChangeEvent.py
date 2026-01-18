class DictChangeEvent:
    """ Event object to represent mutations on a dict.

    Note that the API is different from the ``TraitDictEvent`` emitted via the
    "*name* _items" trait. In particular, the attribute ``changed`` is not
    defined here.

    The interface of this object is provisional as of version 6.1.

    Attributes
    ----------
    object : traits.trait_dict_object.TraitDict
        The dict being mutated.
    removed : dict
        Keys and values for removed or updated items.
        If keys are found in ``added`` as well, they refer to updated items
        and the values are old.
    added : dict
        Keys and values for added or updated items.
        If keys are found in ``removed`` as well, they refer to updated items
        and the values are new.
    """

    def __init__(self, *, object, removed, added):
        self.object = object
        self.removed = removed
        self.added = added

    def __repr__(self):
        return f'{self.__class__.__name__}(object={self.object!r}, removed={self.removed!r}, added={self.added!r})'