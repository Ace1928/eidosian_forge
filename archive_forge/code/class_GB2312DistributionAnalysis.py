from typing import Tuple, Union
from .big5freq import (
from .euckrfreq import (
from .euctwfreq import (
from .gb2312freq import (
from .jisfreq import (
from .johabfreq import JOHAB_TO_EUCKR_ORDER_TABLE
class GB2312DistributionAnalysis(CharDistributionAnalysis):

    def __init__(self) -> None:
        super().__init__()
        self._char_to_freq_order = GB2312_CHAR_TO_FREQ_ORDER
        self._table_size = GB2312_TABLE_SIZE
        self.typical_distribution_ratio = GB2312_TYPICAL_DISTRIBUTION_RATIO

    def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
        first_char, second_char = (byte_str[0], byte_str[1])
        if first_char >= 176 and second_char >= 161:
            return 94 * (first_char - 176) + second_char - 161
        return -1