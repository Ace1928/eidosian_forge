from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def _update_counts(self, *largs):
    pc = self.page_count
    uc = self.up_count
    rc = self.right_count
    sc = self.scroll_count
    self._offset_counts = {'pageup': -pc, 'pagedown': pc, 'up': -uc, 'down': uc, 'right': rc, 'left': -rc, 'scrollup': sc, 'scrolldown': -sc, 'scrollright': -sc, 'scrollleft': sc}