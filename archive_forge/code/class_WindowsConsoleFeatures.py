import sys
from dataclasses import dataclass
@dataclass
class WindowsConsoleFeatures:
    """Windows features available."""
    vt: bool = False
    'The console supports VT codes.'
    truecolor: bool = False
    'The console supports truecolor.'