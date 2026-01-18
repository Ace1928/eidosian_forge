from ..pari import pari
import fractions
def _assert_at_most_one_zero_entry_per_row_or_column(m):
    for i in range(len(m)):
        num_non_zero_entries = 0
        for j in range(len(m[0])):
            if not m[i][j] == 0:
                num_non_zero_entries += 1
        assert num_non_zero_entries < 2
    for j in range(len(m[0])):
        num_non_zero_entries = 0
        for i in range(len(m)):
            if not m[i][j] == 0:
                num_non_zero_entries += 1
        assert num_non_zero_entries < 2