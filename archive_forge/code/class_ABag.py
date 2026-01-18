class ABag:
    """
    'Attribute Bag' - a trivial BAG class for holding attributes.

    This predates modern Python.  Doing this again, we'd use a subclass
    of dict.

    You may initialize with keyword arguments.
    a = ABag(k0=v0,....,kx=vx,....) ==> getattr(a,'kx')==vx

    c = a.clone(ak0=av0,.....) copy with optional additional attributes.
    """

    def __init__(self, **attr):
        self.__dict__.update(attr)

    def clone(self, **attr):
        n = self.__class__(**self.__dict__)
        if attr:
            n.__dict__.update(attr)
        return n

    def __repr__(self):
        D = self.__dict__
        K = list(D.keys())
        K.sort()
        return '%s(%s)' % (self.__class__.__name__, ', '.join(['%s=%r' % (k, D[k]) for k in K]))