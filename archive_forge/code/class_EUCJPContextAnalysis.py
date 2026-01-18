from typing import List, Tuple, Union
class EUCJPContextAnalysis(JapaneseContextAnalysis):

    def get_order(self, byte_str: Union[bytes, bytearray]) -> Tuple[int, int]:
        if not byte_str:
            return (-1, 1)
        first_char = byte_str[0]
        if first_char == 142 or 161 <= first_char <= 254:
            char_len = 2
        elif first_char == 143:
            char_len = 3
        else:
            char_len = 1
        if len(byte_str) > 1:
            second_char = byte_str[1]
            if first_char == 164 and 161 <= second_char <= 243:
                return (second_char - 161, char_len)
        return (-1, char_len)