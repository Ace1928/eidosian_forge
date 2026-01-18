import colorsys
import io
from time import process_time
from pip._vendor.rich import box
from pip._vendor.rich.color import Color
from pip._vendor.rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from pip._vendor.rich.markdown import Markdown
from pip._vendor.rich.measure import Measurement
from pip._vendor.rich.pretty import Pretty
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._vendor.rich.syntax import Syntax
from pip._vendor.rich.table import Table
from pip._vendor.rich.text import Text
def __rich_measure__(self, console: 'Console', options: ConsoleOptions) -> Measurement:
    return Measurement(1, options.max_width)