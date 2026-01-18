from operator import __getitem__
class BaseTestIEnumerableMapping(BaseTestIReadMapping):

    def test_keys(self):
        inst = self._IEnumerableMapping__sample()
        state = self._IEnumerableMapping__stateDict()
        test_keys(self, inst, state)

    def test_values(self):
        inst = self._IEnumerableMapping__sample()
        state = self._IEnumerableMapping__stateDict()
        test_values(self, inst, state)

    def test_items(self):
        inst = self._IEnumerableMapping__sample()
        state = self._IEnumerableMapping__stateDict()
        test_items(self, inst, state)

    def test___len__(self):
        inst = self._IEnumerableMapping__sample()
        state = self._IEnumerableMapping__stateDict()
        test___len__(self, inst, state)

    def _IReadMapping__stateDict(self):
        return self._IEnumerableMapping__stateDict()

    def _IReadMapping__sample(self):
        return self._IEnumerableMapping__sample()

    def _IReadMapping__absentKeys(self):
        return self._IEnumerableMapping__absentKeys()