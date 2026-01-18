from ._base import DirectiveParser, BaseDirective, DirectivePlugin
from ._rst import RSTDirective
from ._fenced import FencedDirective
from .admonition import Admonition
from .toc import TableOfContents
from .include import Include
from .image import Image, Figure
class RstDirective(RSTDirective):

    def __init__(self, plugins):
        super(RstDirective, self).__init__(plugins)
        import warnings
        warnings.warn("'RstDirective' is deprecated, please use 'RSTDirective' instead.", DeprecationWarning, stacklevel=2)