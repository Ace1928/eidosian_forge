from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class NumberFormatter(StringFormatter):
    """ Number cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    format = String('0,0', help="\n    The number format, as defined in the following tables:\n\n    **NUMBERS**:\n\n    ============ ============== ===============\n    Number       Format         String\n    ============ ============== ===============\n    10000        '0,0.0000'     10,000.0000\n    10000.23     '0,0'          10,000\n    10000.23     '+0,0'         +10,000\n    -10000       '0,0.0'        -10,000.0\n    10000.1234   '0.000'        10000.123\n    10000.1234   '0[.]00000'    10000.12340\n    -10000       '(0,0.0000)'   (10,000.0000)\n    -0.23        '.00'          -.23\n    -0.23        '(.00)'        (.23)\n    0.23         '0.00000'      0.23000\n    0.23         '0.0[0000]'    0.23\n    1230974      '0.0a'         1.2m\n    1460         '0 a'          1 k\n    -104000      '0a'           -104k\n    1            '0o'           1st\n    52           '0o'           52nd\n    23           '0o'           23rd\n    100          '0o'           100th\n    ============ ============== ===============\n\n    **CURRENCY**:\n\n    =========== =============== =============\n    Number      Format          String\n    =========== =============== =============\n    1000.234    '$0,0.00'       $1,000.23\n    1000.2      '0,0[.]00 $'    1,000.20 $\n    1001        '$ 0,0[.]00'    $ 1,001\n    -1000.234   '($0,0)'        ($1,000)\n    -1000.234   '$0.00'         -$1000.23\n    1230974     '($ 0.00 a)'    $ 1.23 m\n    =========== =============== =============\n\n    **BYTES**:\n\n    =============== =========== ============\n    Number          Format      String\n    =============== =========== ============\n    100             '0b'        100B\n    2048            '0 b'       2 KB\n    7884486213      '0.0b'      7.3GB\n    3467479682787   '0.000 b'   3.154 TB\n    =============== =========== ============\n\n    **PERCENTAGES**:\n\n    ============= ============= ===========\n    Number        Format        String\n    ============= ============= ===========\n    1             '0%'          100%\n    0.974878234   '0.000%'      97.488%\n    -0.43         '0 %'         -43 %\n    0.43          '(0.000 %)'   43.000 %\n    ============= ============= ===========\n\n    **TIME**:\n\n    ============ ============== ============\n    Number       Format         String\n    ============ ============== ============\n    25           '00:00:00'     0:00:25\n    238          '00:00:00'     0:03:58\n    63846        '00:00:00'     17:44:06\n    ============ ============== ============\n\n    For the complete specification, see http://numbrojs.com/format.html\n    ")
    language = Enum(NumeralLanguage, default='en', help='\n    The language to use for formatting language-specific features (e.g. thousands separator).\n    ')
    rounding = Enum(RoundingFunction, help='\n    Rounding functions (round, floor, ceil) and their synonyms (nearest, rounddown, roundup).\n    ')