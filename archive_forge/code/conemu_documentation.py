from __future__ import annotations
import sys
from typing import Any, TextIO
from prompt_toolkit.data_structures import Size
from .base import Output
from .color_depth import ColorDepth
from .vt100 import Vt100_Output
from .win32 import Win32Output

    ConEmu (Windows) output abstraction.

    ConEmu is a Windows console application, but it also supports ANSI escape
    sequences. This output class is actually a proxy to both `Win32Output` and
    `Vt100_Output`. It uses `Win32Output` for console sizing and scrolling, but
    all cursor movements and scrolling happens through the `Vt100_Output`.

    This way, we can have 256 colors in ConEmu and Cmder. Rendering will be
    even a little faster as well.

    http://conemu.github.io/
    http://gooseberrycreative.com/cmder/
    