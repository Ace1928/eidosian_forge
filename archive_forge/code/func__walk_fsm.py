from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
@numba.njit(nogil=True, cache=True)
def _walk_fsm(fsm_transitions: Dict[Tuple[int, int], int], alphabet_symbol_mapping: Dict[str, int], alphabet_anything_value: int, fsm_initial: int, fsm_finals: Set[int], input_string: str, start_state: int, full_match: bool=True) -> List[int]:
    state = start_state
    accepted_states: List[int] = numba.typed.List.empty_list(numba.int64)
    last_final_idx: int = numba.uint64(0)
    for i, symbol in enumerate(input_string):
        trans_key = alphabet_symbol_mapping.get(symbol, alphabet_anything_value)
        new_state = fsm_transitions.get((state, trans_key))
        if new_state is None:
            if not full_match and last_final_idx > 0:
                return accepted_states[:last_final_idx]
            return numba.typed.List.empty_list(numba.int64)
        state = new_state
        if state in fsm_finals:
            last_final_idx = numba.uint64(i + 1)
        accepted_states.append(_nonoptional(state))
    if full_match and last_final_idx - 1 != i:
        return numba.typed.List.empty_list(numba.int64)
    return accepted_states