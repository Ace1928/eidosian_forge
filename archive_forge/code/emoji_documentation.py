import sys
from typing import TYPE_CHECKING, Optional, Union
from .jupyter import JupyterMixin
from .segment import Segment
from .style import Style
from ._emoji_codes import EMOJI
from ._emoji_replace import _emoji_replace
Replace emoji markup with corresponding unicode characters.

        Args:
            text (str): A string with emojis codes, e.g. "Hello :smiley:!"

        Returns:
            str: A string with emoji codes replaces with actual emoji.
        