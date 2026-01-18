import gtk, gobject
def _main_quit(*a, **kw):
    gtk.main_quit()
    return False