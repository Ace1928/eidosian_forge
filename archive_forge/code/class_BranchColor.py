import collections
import copy
import itertools
import random
import re
import warnings
class BranchColor:
    """Indicates the color of a clade when rendered graphically.

    The color should be interpreted by client code (e.g. visualization
    programs) as applying to the whole clade, unless overwritten by the
    color(s) of sub-clades.

    Color values must be integers from 0 to 255.
    """
    color_names = {'red': (255, 0, 0), 'r': (255, 0, 0), 'yellow': (255, 255, 0), 'y': (255, 255, 0), 'green': (0, 128, 0), 'g': (0, 128, 0), 'cyan': (0, 255, 255), 'c': (0, 255, 255), 'blue': (0, 0, 255), 'b': (0, 0, 255), 'magenta': (255, 0, 255), 'm': (255, 0, 255), 'black': (0, 0, 0), 'k': (0, 0, 0), 'white': (255, 255, 255), 'w': (255, 255, 255), 'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'lime': (0, 255, 0), 'aqua': (0, 255, 255), 'teal': (0, 128, 128), 'navy': (0, 0, 128), 'fuchsia': (255, 0, 255), 'purple': (128, 0, 128), 'silver': (192, 192, 192), 'gray': (128, 128, 128), 'grey': (128, 128, 128), 'pink': (255, 192, 203), 'salmon': (250, 128, 114), 'orange': (255, 165, 0), 'gold': (255, 215, 0), 'tan': (210, 180, 140), 'brown': (165, 42, 42)}

    def __init__(self, red, green, blue):
        """Initialize BranchColor for a tree."""
        for color in (red, green, blue):
            assert isinstance(color, int) and 0 <= color <= 255, 'Color values must be integers between 0 and 255.'
        self.red = red
        self.green = green
        self.blue = blue

    @classmethod
    def from_hex(cls, hexstr):
        """Construct a BranchColor object from a hexadecimal string.

        The string format is the same style used in HTML and CSS, such as
        '#FF8000' for an RGB value of (255, 128, 0).
        """
        assert isinstance(hexstr, str) and hexstr.startswith('#') and (len(hexstr) == 7), 'need a 24-bit hexadecimal string, e.g. #000000'
        RGB = (hexstr[1:3], hexstr[3:5], hexstr[5:])
        return cls(*(int('0x' + cc, base=16) for cc in RGB))

    @classmethod
    def from_name(cls, colorname):
        """Construct a BranchColor object by the color's name."""
        return cls(*cls.color_names[colorname])

    def to_hex(self):
        """Return a 24-bit hexadecimal RGB representation of this color.

        The returned string is suitable for use in HTML/CSS, as a color
        parameter in matplotlib, and perhaps other situations.

        Examples
        --------
        >>> bc = BranchColor(12, 200, 100)
        >>> bc.to_hex()
        '#0cc864'

        """
        return f'#{self.red:02x}{self.green:02x}{self.blue:02x}'

    def to_rgb(self):
        """Return a tuple of RGB values (0 to 255) representing this color.

        Examples
        --------
        >>> bc = BranchColor(255, 165, 0)
        >>> bc.to_rgb()
        (255, 165, 0)

        """
        return (self.red, self.green, self.blue)

    def __repr__(self) -> str:
        """Preserve the standard RGB order when representing this object."""
        return '%s(red=%d, green=%d, blue=%d)' % (self.__class__.__name__, self.red, self.green, self.blue)

    def __str__(self) -> str:
        """Show the color's RGB values."""
        return '(%d, %d, %d)' % (self.red, self.green, self.blue)