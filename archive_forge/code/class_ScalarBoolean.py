from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.anchor import Anchor
class ScalarBoolean(int):

    def __new__(cls, *args, **kw):
        anchor = kw.pop('anchor', None)
        b = int.__new__(cls, *args, **kw)
        if anchor is not None:
            b.yaml_set_anchor(anchor, always_dump=True)
        return b

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