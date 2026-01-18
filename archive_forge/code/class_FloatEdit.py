from __future__ import annotations
import re
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING
from urwid import Edit
class FloatEdit(NumEdit):
    """Edit widget for float values."""

    def __init__(self, caption='', default: str | int | Decimal | None=None, preserveSignificance: bool | None=None, decimalSeparator: str | None=None, *, preserve_significance: bool=True, decimal_separator: str='.', allow_negative: bool=False) -> None:
        """
        caption -- caption markup
        default -- default edit value
        preserve_significance -- return value has the same signif. as default
        decimal_separator -- use '.' as separator by default, optionally a ','

        >>> FloatEdit(u"",  "1.065434")
        <FloatEdit selectable flow widget '1.065434' edit_pos=8>
        >>> e, size = FloatEdit(u"", "1.065434"), (10,)
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> assert e.edit_text == ".065434"
        >>> e.keypress(size, 'end')
        >>> e.keypress(size, 'backspace')
        >>> assert e.edit_text == ".06543"
        >>> e, size = FloatEdit(), (10,)
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> e.keypress(size, '.')
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> assert e.value() == Decimal("51.51"), e.value()
        >>> e, size = FloatEdit(decimal_separator=":"), (10,)
        Traceback (most recent call last):
            ...
        ValueError: invalid decimal separator: :
        >>> e, size = FloatEdit(decimal_separator=","), (10,)
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> e.keypress(size, ',')
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> assert e.edit_text == "51,51"
        >>> e, size = FloatEdit("", "3.1415", preserve_significance=True), (10,)
        >>> e.keypress(size, 'end')
        >>> e.keypress(size, 'backspace')
        >>> e.keypress(size, 'backspace')
        >>> assert e.edit_text == "3.14"
        >>> assert e.value() == Decimal("3.1400")
        >>> e.keypress(size, '1')
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '9')
        >>> assert e.value() == Decimal("3.1416"), e.value()
        >>> e, size = FloatEdit("", ""), (10,)
        >>> assert e.value() is None
        >>> e, size = FloatEdit(u"", 10.0), (10,)
        Traceback (most recent call last):
            ...
        ValueError: default: Only 'str', 'int', 'long' or Decimal input allowed
        """
        self.significance = None
        self._decimal_separator = decimal_separator
        if decimalSeparator is not None:
            warnings.warn("'decimalSeparator' argument is deprecated. Use 'decimal_separator' keyword argument", DeprecationWarning, stacklevel=3)
            self._decimal_separator = decimalSeparator
        if self._decimal_separator not in {'.', ','}:
            raise ValueError(f'invalid decimal separator: {self._decimal_separator}')
        if preserveSignificance is not None:
            warnings.warn("'preserveSignificance' argument is deprecated. Use 'preserve_significance' keyword argument", DeprecationWarning, stacklevel=3)
            preserve_significance = preserveSignificance
        val = ''
        if default is not None and default != '':
            if not isinstance(default, (int, str, Decimal)):
                raise ValueError("default: Only 'str', 'int' or Decimal input allowed")
            if isinstance(default, str) and default:
                float(default)
                default = Decimal(default)
            if preserve_significance and isinstance(default, Decimal):
                self.significance = default
            val = str(default)
        super().__init__(self.ALLOWED[0:10] + self._decimal_separator, caption, val, allow_negative=allow_negative)

    def value(self) -> Decimal | None:
        """
        Return the numeric value of self.edit_text.
        """
        if self.edit_text:
            normalized = Decimal(self.edit_text.replace(self._decimal_separator, '.'))
            if self.significance is not None:
                return normalized.quantize(self.significance)
            return normalized
        return None

    def __float__(self) -> float:
        """Enforced float value return.

        >>> e, size = FloatEdit(allow_negative=True), (10,)
        >>> assert float(e) == 0.
        >>> e.keypress(size, '-')
        >>> e.keypress(size, '4')
        >>> e.keypress(size, '.')
        >>> e.keypress(size, '2')
        >>> assert float(e) == -4.2
        """
        if self.edit_text:
            return float(self.edit_text.replace(self._decimal_separator, '.'))
        return 0.0