import abc
def _items_iter(self):
    """Iterator for each item returned."""
    for page in self._page_iter(increment=False):
        for item in page:
            self.num_results += 1
            yield item