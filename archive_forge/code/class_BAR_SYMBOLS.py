from __future__ import annotations
import dataclasses
import enum
import typing
class BAR_SYMBOLS(str, enum.Enum):
    """Standard Unicode bar symbols excluding empty space.

    Start from space (0), then 1/8 till full block (1/1).
    Typically used only 8 from this symbol collection depends on use-case:
    * empty - 7/8 and styles for BG different on both sides (like standard `ProgressBar` and `BarGraph`)
    * 1/8 - full block and single style for BG on the right side
    """
    HORISONTAL = ' ▏▎▍▌▋▊▉█'
    VERTICAL = ' ▁▂▃▄▅▆▇█'