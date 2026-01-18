import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
class _NumberRangeBase(_NumberParamTypeBase):

    def __init__(self, min: t.Optional[float]=None, max: t.Optional[float]=None, min_open: bool=False, max_open: bool=False, clamp: bool=False) -> None:
        self.min = min
        self.max = max
        self.min_open = min_open
        self.max_open = max_open
        self.clamp = clamp

    def to_info_dict(self) -> t.Dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(min=self.min, max=self.max, min_open=self.min_open, max_open=self.max_open, clamp=self.clamp)
        return info_dict

    def convert(self, value: t.Any, param: t.Optional['Parameter'], ctx: t.Optional['Context']) -> t.Any:
        import operator
        rv = super().convert(value, param, ctx)
        lt_min: bool = self.min is not None and (operator.le if self.min_open else operator.lt)(rv, self.min)
        gt_max: bool = self.max is not None and (operator.ge if self.max_open else operator.gt)(rv, self.max)
        if self.clamp:
            if lt_min:
                return self._clamp(self.min, 1, self.min_open)
            if gt_max:
                return self._clamp(self.max, -1, self.max_open)
        if lt_min or gt_max:
            self.fail(_('{value} is not in the range {range}.').format(value=rv, range=self._describe_range()), param, ctx)
        return rv

    def _clamp(self, bound: float, dir: 'te.Literal[1, -1]', open: bool) -> float:
        """Find the valid value to clamp to bound in the given
        direction.

        :param bound: The boundary value.
        :param dir: 1 or -1 indicating the direction to move.
        :param open: If true, the range does not include the bound.
        """
        raise NotImplementedError

    def _describe_range(self) -> str:
        """Describe the range for use in help text."""
        if self.min is None:
            op = '<' if self.max_open else '<='
            return f'x{op}{self.max}'
        if self.max is None:
            op = '>' if self.min_open else '>='
            return f'x{op}{self.min}'
        lop = '<' if self.min_open else '<='
        rop = '<' if self.max_open else '<='
        return f'{self.min}{lop}x{rop}{self.max}'

    def __repr__(self) -> str:
        clamp = ' clamped' if self.clamp else ''
        return f'<{type(self).__name__} {self._describe_range()}{clamp}>'