class AuthContext(Mapping):
    """ Holds a set of keys and values relevant to authorization.

    It is passed as an argument to authorization checkers, so that the checkers
    can access information about the context of the authorization request.
    It is immutable - values can only be added by copying the whole thing.
    """

    def __init__(self, somedict=None):
        if somedict is None:
            somedict = {}
        self._dict = dict(somedict)
        self._hash = None

    def with_value(self, key, val):
        """ Return a copy of the AuthContext object with the given key and
        value added.
        """
        new_dict = dict(self._dict)
        new_dict[key] = val
        return AuthContext(new_dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        return self._dict == other._dict