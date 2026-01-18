from collections.abc import Mapping
class FilterMultiInner(FilterAdjacency):

    def __iter__(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            my_nodes = (n for n in self.NODE_OK.nodes if n in self._atlas)
        else:
            my_nodes = (n for n in self._atlas if self.NODE_OK(n))
        for n in my_nodes:
            some_keys_ok = False
            for key in self._atlas[n]:
                if self.EDGE_OK(n, key):
                    some_keys_ok = True
                    break
            if some_keys_ok is True:
                yield n

    def __getitem__(self, nbr):
        if nbr in self._atlas and self.NODE_OK(nbr):

            def new_node_ok(key):
                return self.EDGE_OK(nbr, key)
            return FilterAtlas(self._atlas[nbr], new_node_ok)
        raise KeyError(f'Key {nbr} not found')