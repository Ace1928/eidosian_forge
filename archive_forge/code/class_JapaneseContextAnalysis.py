from typing import List, Tuple, Union
class JapaneseContextAnalysis:
    NUM_OF_CATEGORY = 6
    DONT_KNOW = -1
    ENOUGH_REL_THRESHOLD = 100
    MAX_REL_THRESHOLD = 1000
    MINIMUM_DATA_THRESHOLD = 4

    def __init__(self) -> None:
        self._total_rel = 0
        self._rel_sample: List[int] = []
        self._need_to_skip_char_num = 0
        self._last_char_order = -1
        self._done = False
        self.reset()

    def reset(self) -> None:
        self._total_rel = 0
        self._rel_sample = [0] * self.NUM_OF_CATEGORY
        self._need_to_skip_char_num = 0
        self._last_char_order = -1
        self._done = False

    def feed(self, byte_str: Union[bytes, bytearray], num_bytes: int) -> None:
        if self._done:
            return
        i = self._need_to_skip_char_num
        while i < num_bytes:
            order, char_len = self.get_order(byte_str[i:i + 2])
            i += char_len
            if i > num_bytes:
                self._need_to_skip_char_num = i - num_bytes
                self._last_char_order = -1
            else:
                if order != -1 and self._last_char_order != -1:
                    self._total_rel += 1
                    if self._total_rel > self.MAX_REL_THRESHOLD:
                        self._done = True
                        break
                    self._rel_sample[jp2_char_context[self._last_char_order][order]] += 1
                self._last_char_order = order

    def got_enough_data(self) -> bool:
        return self._total_rel > self.ENOUGH_REL_THRESHOLD

    def get_confidence(self) -> float:
        if self._total_rel > self.MINIMUM_DATA_THRESHOLD:
            return (self._total_rel - self._rel_sample[0]) / self._total_rel
        return self.DONT_KNOW

    def get_order(self, _: Union[bytes, bytearray]) -> Tuple[int, int]:
        return (-1, 1)