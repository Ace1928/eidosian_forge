from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class Pos(NamedTuple):
    adv: int
    dx: int
    dy: int
    adv_adjust_by: dict
    dx_adjust_by: dict
    dy_adjust_by: dict

    def __str__(self):
        res = ' POS'
        for attr in ('adv', 'dx', 'dy'):
            value = getattr(self, attr)
            if value is not None:
                res += f' {attr.upper()} {value}'
                adjust_by = getattr(self, f'{attr}_adjust_by', {})
                for size, adjustment in adjust_by.items():
                    res += f' ADJUST_BY {adjustment} AT {size}'
        res += ' END_POS'
        return res