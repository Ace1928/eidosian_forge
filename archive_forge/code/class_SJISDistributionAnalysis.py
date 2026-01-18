from typing import Tuple, Union
from .big5freq import (
from .euckrfreq import (
from .euctwfreq import (
from .gb2312freq import (
from .jisfreq import (
from .johabfreq import JOHAB_TO_EUCKR_ORDER_TABLE
class SJISDistributionAnalysis(CharDistributionAnalysis):

    def __init__(self) -> None:
        super().__init__()
        self._char_to_freq_order = JIS_CHAR_TO_FREQ_ORDER
        self._table_size = JIS_TABLE_SIZE
        self.typical_distribution_ratio = JIS_TYPICAL_DISTRIBUTION_RATIO

    def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
        first_char, second_char = (byte_str[0], byte_str[1])
        if 129 <= first_char <= 159:
            order = 188 * (first_char - 129)
        elif 224 <= first_char <= 239:
            order = 188 * (first_char - 224 + 31)
        else:
            return -1
        order = order + second_char - 64
        if second_char > 127:
            order = -1
        return order