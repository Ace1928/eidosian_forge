import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def _str_full_(self) -> str:
    string = ''
    string += 'stable' + ' ' * max(self.n * 2 - 3, 1)
    string += '| destable\n'
    string += '-' * max(7, self.n * 2 + 3) + '+' + '-' * max(10, self.n * 2 + 4) + '\n'
    for j in range(self.n):
        for i in [j + self.n, j]:
            string += '- ' if self.rs[i] else '+ '
            for k in range(self.n):
                if self.xs[i, k] & (not self.zs[i, k]):
                    string += f'X{k}'
                elif (not self.xs[i, k]) & self.zs[i, k]:
                    string += f'Z{k}'
                elif self.xs[i, k] & self.zs[i, k]:
                    string += f'Y{k}'
                else:
                    string += '  '
            if i == j + self.n:
                string += ' ' * max(0, 4 - self.n * 2) + ' | '
        string += '\n'
    return string