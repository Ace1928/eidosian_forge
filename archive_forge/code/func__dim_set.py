def _dim_set(positional, arg):

    def convert(a):
        if isinstance(a, Dim):
            return a
        else:
            assert isinstance(a, int)
            return positional[a]
    if arg is None:
        return positional
    elif not isinstance(arg, (Dim, int)):
        return tuple((convert(a) for a in arg))
    else:
        return (convert(arg),)