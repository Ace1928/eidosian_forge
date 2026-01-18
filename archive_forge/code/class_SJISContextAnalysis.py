from typing import List, Tuple, Union
class SJISContextAnalysis(JapaneseContextAnalysis):

    def __init__(self) -> None:
        super().__init__()
        self._charset_name = 'SHIFT_JIS'

    @property
    def charset_name(self) -> str:
        return self._charset_name

    def get_order(self, byte_str: Union[bytes, bytearray]) -> Tuple[int, int]:
        if not byte_str:
            return (-1, 1)
        first_char = byte_str[0]
        if 129 <= first_char <= 159 or 224 <= first_char <= 252:
            char_len = 2
            if first_char == 135 or 250 <= first_char <= 252:
                self._charset_name = 'CP932'
        else:
            char_len = 1
        if len(byte_str) > 1:
            second_char = byte_str[1]
            if first_char == 202 and 159 <= second_char <= 241:
                return (second_char - 159, char_len)
        return (-1, char_len)