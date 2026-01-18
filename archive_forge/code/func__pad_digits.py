import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from cirq import protocols
from cirq.ops import raw_types
def _pad_digits(text: str) -> str:
    """A str method with hacks to support better lexicographic ordering.

    The output strings are not intended to be human readable.

    The returned string will have digit-runs zero-padded up to at least 8
    digits. That way, instead of 'a10' coming before 'a2', 'a000010' will come
    after 'a000002'.

    Also, the original length of each digit-run is appended after the
    zero-padded run. This is so that 'a0' continues to come before 'a00'.
    """
    was_on_digits = False
    last_transition = 0
    chunks = []

    def handle_transition_at(k):
        chunk = text[last_transition:k]
        if was_on_digits:
            chunk = chunk.rjust(8, '0') + ':' + str(len(chunk))
        chunks.append(chunk)
    for i in range(len(text)):
        on_digits = text[i].isdigit()
        if was_on_digits != on_digits:
            handle_transition_at(i)
            was_on_digits = on_digits
            last_transition = i
    handle_transition_at(len(text))
    return ''.join(chunks)