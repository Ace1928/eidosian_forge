import array
from typing import (
class CCITTFaxDecoder(CCITTG4Parser):

    def __init__(self, width: int, bytealign: bool=False, reversed: bool=False) -> None:
        CCITTG4Parser.__init__(self, width, bytealign=bytealign)
        self.reversed = reversed
        self._buf = b''
        return

    def close(self) -> bytes:
        return self._buf

    def output_line(self, y: int, bits: Sequence[int]) -> None:
        arr = array.array('B', [0] * ((len(bits) + 7) // 8))
        if self.reversed:
            bits = [1 - b for b in bits]
        for i, b in enumerate(bits):
            if b:
                arr[i // 8] += (128, 64, 32, 16, 8, 4, 2, 1)[i % 8]
        self._buf += arr.tobytes()
        return