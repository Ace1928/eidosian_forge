from typing import NamedTuple
from typing import Sequence, Tuple
def __all_tuple_init__(self, shots: Sequence[Tuple]):
    res = []
    total_shots = 0
    current_shots, current_copies = shots[0]
    for s in shots[1:]:
        if s[0] == current_shots:
            current_copies += s[1]
        else:
            res.append(ShotCopies(current_shots, current_copies))
            total_shots += current_shots * current_copies
            current_shots, current_copies = s
    self.shot_vector = tuple(res + [ShotCopies(current_shots, current_copies)])
    self.total_shots = total_shots + current_shots * current_copies