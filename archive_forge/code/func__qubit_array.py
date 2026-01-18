import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _qubit_array(reg: Register):
    qubits = np.empty(reg.shape + (reg.bitsize,), dtype=object)
    for ii in reg.all_idxs():
        for j in range(reg.bitsize):
            prefix = '' if not ii else f'[{', '.join((str(i) for i in ii))}]'
            suffix = '' if reg.bitsize == 1 else f'[{j}]'
            qubits[ii + (j,)] = cirq.NamedQubit(reg.name + prefix + suffix)
    return qubits