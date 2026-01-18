class _StaticMRO:
    had_inconsistency = None

    def __init__(self, C, mro):
        self.leaf = C
        self.__mro = tuple(mro)

    def mro(self):
        return list(self.__mro)