class SetChangeEvent:
    """ Event object to represent mutations on a set.

    The interface of this object is provisional as of version 6.1.

    Attributes
    ----------
    object : traits.trait_set_object.TraitSet
        The set being mutated.
    removed : set
        Values removed from the set.
    added : set
        Values added to the set.
    """

    def __init__(self, *, object, removed, added):
        self.object = object
        self.removed = removed
        self.added = added

    def __repr__(self):
        return f'{self.__class__.__name__}(object={self.object!r}, removed={self.removed!r}, added={self.added!r})'