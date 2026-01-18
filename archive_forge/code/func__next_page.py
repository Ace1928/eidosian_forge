def _next_page(self):
    for link in self._links:
        if link['rel'] == 'next':
            iterables = self._client.follow(link['href'])
            if iterables:
                self.get_iterables(iterables)
                return
    raise StopIteration