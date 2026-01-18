import math
def _show(self):
    print('=' * 20)
    for b in range(self._buckets):
        b = (b + self._buckets_index) % self._buckets
        vals = [self._bucket[b][i] for i in range(self._index[b])]
        print(f'{b}: {vals}')