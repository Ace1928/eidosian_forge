import re
def _lists_with_fixed_sum_iterator(N, l):
    if l == 1:
        yield [N]
    else:
        for i in range(N + 1):
            for j in _lists_with_fixed_sum_iterator(N - i, l - 1):
                yield ([i] + j)