class TraitChangeEvent:
    """ Emitted when a trait on a HasTraits object is changed.

    The interface of this object is provisional as of version 6.1.

    Attributes
    ----------
    object : HasTraits
        Object on which a trait is changed.
    name : str
        Name of the trait.
    old : any
        The old value.
    new : any
        The new value.
    """

    def __init__(self, *, object, name, old, new):
        self.object = object
        self.name = name
        self.old = old
        self.new = new

    def __repr__(self):
        return '{event.__class__.__name__}(object={event.object!r}, name={event.name!r}, old={event.old!r}, new={event.new!r})'.format(event=self)