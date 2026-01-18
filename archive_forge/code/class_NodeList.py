import xml.dom
class NodeList(list):
    __slots__ = ()

    def item(self, index):
        if 0 <= index < len(self):
            return self[index]

    def _get_length(self):
        return len(self)

    def _set_length(self, value):
        raise xml.dom.NoModificationAllowedErr("attempt to modify read-only attribute 'length'")
    length = property(_get_length, _set_length, doc='The number of nodes in the NodeList.')

    def __setstate__(self, state):
        if state is None:
            state = []
        self[:] = state