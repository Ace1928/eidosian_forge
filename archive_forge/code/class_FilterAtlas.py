from collections.abc import Mapping
class FilterAtlas(Mapping):

    def __init__(self, d, NODE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK

    def __len__(self):
        return sum((1 for n in self))

    def __iter__(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, key):
        if key in self._atlas and self.NODE_OK(key):
            return self._atlas[key]
        raise KeyError(f'Key {key} not found')

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f'{self.__class__.__name__}({self._atlas!r}, {self.NODE_OK!r})'