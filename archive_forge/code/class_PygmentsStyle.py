from __future__ import unicode_literals
from .base import Style
from .from_dict import style_from_dict
class PygmentsStyle(Style):
    """ Deprecated. """

    def __new__(cls, pygments_style_cls):
        assert issubclass(pygments_style_cls, pygments_Style)
        return style_from_dict(pygments_style_cls.styles)

    def invalidation_hash(self):
        pass

    @classmethod
    def from_defaults(cls, style_dict=None, pygments_style_cls=pygments_DefaultStyle, include_extensions=True):
        """ Deprecated. """
        return style_from_pygments(style_cls=pygments_style_cls, style_dict=style_dict, include_defaults=include_extensions)