from __future__ import annotations
import dataclasses
import enum
import typing
class _BoxSymbolsCollection(typing.NamedTuple):
    """Standard Unicode box symbols for basic tables drawing.

    .. note::
        Transitions are not included: depends on line types, different kinds of transitions are available.
        Please check Unicode table for transitions symbols if required.
    """
    LIGHT: _LightBoxSymbols = _LightBoxSymbols('─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '┈', '┄', '╌', '╎', '┆', '┊', '╭', '╮', '╰', '╯')
    HEAVY: _BoxSymbolsWithDashes = _BoxSymbolsWithDashes('━', '┃', '┏', '┓', '┗', '┛', '┣', '┫', '┳', '┻', '╋', '┉', '┅', '╍', '╏', '┇', '┋')
    DOUBLE: _BoxSymbols = _BoxSymbols('═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╦', '╩', '╬')