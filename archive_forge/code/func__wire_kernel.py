import sys
import warnings
import gobject
import gtk
def _wire_kernel(self):
    """Initializes the kernel inside GTK.

        This is meant to run only once at startup, so it does its job and
        returns False to ensure it doesn't get run again by GTK.
        """
    self.gtk_main, self.gtk_main_quit = self._hijack_gtk()
    gobject.timeout_add(int(1000 * self.kernel._poll_interval), self.iterate_kernel)
    return False