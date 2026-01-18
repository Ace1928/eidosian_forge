class _NamedBool(int):

    def __new__(cls, val, name):
        inst = super(cls, _NamedBool).__new__(cls, val)
        inst.__name__ = name
        return inst