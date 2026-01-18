import typing as t
class SummaryItem:
    """Analogous to the SummaryItem protobuf message."""
    key: t.Tuple[str]
    value: t.Any

    def __init__(self):
        self.key = tuple()
        self.value = None

    def __str__(self):
        return 'SummaryItem: key: ' + str(self.key) + ' value: ' + str(self.value)
    __repr__ = __str__

    def _add_next_parent(self, parent_key):
        with_next_parent = SummaryItem()
        key = self.key
        if not isinstance(key, tuple):
            key = (key,)
        with_next_parent.key = (parent_key,) + self.key
        with_next_parent.value = self.value
        return with_next_parent