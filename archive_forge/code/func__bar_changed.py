import unittest
from traits.api import HasTraits, Str, Instance, Any
def _bar_changed(self, obj, old, new):
    if old is not None and old is not new:
        old.on_trait_change(self._effect_changed, name='effect', remove=True)
        old.foo.on_trait_change(self._cause_changed, name='cause', remove=True)
    if new is not None:
        new.foo.on_trait_change(self._cause_changed, name='cause')
        new.on_trait_change(self._effect_changed, name='effect')