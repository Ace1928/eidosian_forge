import gtk
def _enableButtons(self):
    for w in self._conditional_widgets:
        w.set_sensitive(True)