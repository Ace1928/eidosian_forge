class ListChangeEvent:
    """ Event object to represent mutations to a list.

    The interface of this object is provisional as of version 6.1.

    Attributes
    ----------
    object : traits.trait_list_object.TraitList
        The list being mutated.
    index : int or slice
        The index used for the mutation.
    added : list
        Values added to the list.
    removed : list
        Values removed from the list.
    """

    def __init__(self, *, object, index, removed, added):
        self.object = object
        self.added = added
        self.removed = removed
        self.index = index

    def __repr__(self):
        return f'{self.__class__.__name__}(object={self.object!r}, index={self.index!r}, removed={self.removed!r}, added={self.added!r})'