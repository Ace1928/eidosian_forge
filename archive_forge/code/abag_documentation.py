
    'Attribute Bag' - a trivial BAG class for holding attributes.

    This predates modern Python.  Doing this again, we'd use a subclass
    of dict.

    You may initialize with keyword arguments.
    a = ABag(k0=v0,....,kx=vx,....) ==> getattr(a,'kx')==vx

    c = a.clone(ak0=av0,.....) copy with optional additional attributes.
    