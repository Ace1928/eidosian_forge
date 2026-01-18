from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor
class ScalarString(text_type):
    __slots__ = Anchor.attrib

    def __new__(cls, *args, **kw):
        anchor = kw.pop('anchor', None)
        ret_val = text_type.__new__(cls, *args, **kw)
        if anchor is not None:
            ret_val.yaml_set_anchor(anchor, always_dump=True)
        return ret_val

    def replace(self, old, new, maxreplace=-1):
        return type(self)(text_type.replace(self, old, new, maxreplace))

    @property
    def anchor(self):
        if not hasattr(self, Anchor.attrib):
            setattr(self, Anchor.attrib, Anchor())
        return getattr(self, Anchor.attrib)

    def yaml_anchor(self, any=False):
        if not hasattr(self, Anchor.attrib):
            return None
        if any or self.anchor.always_dump:
            return self.anchor
        return None

    def yaml_set_anchor(self, value, always_dump=False):
        self.anchor.value = value
        self.anchor.always_dump = always_dump