class BatchGetResultSet(ResultSet):

    def __init__(self, *args, **kwargs):
        self._keys_left = kwargs.pop('keys', [])
        self._max_batch_get = kwargs.pop('max_batch_get', 100)
        super(BatchGetResultSet, self).__init__(*args, **kwargs)

    def fetch_more(self):
        self._reset()
        args = self.call_args[:]
        kwargs = self.call_kwargs.copy()
        kwargs['keys'] = self._keys_left[:self._max_batch_get]
        self._keys_left = self._keys_left[self._max_batch_get:]
        if len(self._keys_left) <= 0:
            self._results_left = False
        results = self.the_callable(*args, **kwargs)
        if not len(results.get('results', [])):
            return
        self._results.extend(results['results'])
        for offset, key_data in enumerate(results.get('unprocessed_keys', [])):
            self._keys_left.insert(offset, key_data)
        if len(self._keys_left) > 0:
            self._results_left = True
        if self.call_kwargs.get('limit'):
            self.call_kwargs['limit'] -= len(results['results'])