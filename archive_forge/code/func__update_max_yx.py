import curses
import curses.ascii
def _update_max_yx(self):
    maxy, maxx = self.win.getmaxyx()
    self.maxy = maxy - 1
    self.maxx = maxx - 1