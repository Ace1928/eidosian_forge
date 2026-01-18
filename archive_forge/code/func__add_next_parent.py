import typing as t
def _add_next_parent(self, parent_key):
    with_next_parent = SummaryItem()
    key = self.key
    if not isinstance(key, tuple):
        key = (key,)
    with_next_parent.key = (parent_key,) + self.key
    with_next_parent.value = self.value
    return with_next_parent