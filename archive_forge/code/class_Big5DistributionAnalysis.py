from typing import Tuple, Union
from .big5freq import (
from .euckrfreq import (
from .euctwfreq import (
from .gb2312freq import (
from .jisfreq import (
from .johabfreq import JOHAB_TO_EUCKR_ORDER_TABLE
class Big5DistributionAnalysis(CharDistributionAnalysis):

    def __init__(self) -> None:
        super().__init__()
        self._char_to_freq_order = BIG5_CHAR_TO_FREQ_ORDER
        self._table_size = BIG5_TABLE_SIZE
        self.typical_distribution_ratio = BIG5_TYPICAL_DISTRIBUTION_RATIO

    def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
        first_char, second_char = (byte_str[0], byte_str[1])
        if first_char >= 164:
            if second_char >= 161:
                return 157 * (first_char - 164) + second_char - 161 + 63
            return 157 * (first_char - 164) + second_char - 64
        return -1