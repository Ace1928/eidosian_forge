import pickle
import sys
import numpy as np
from ase.gui.i18n import _
import ase.gui.ui as ui
class Graphs:

    def __init__(self, gui):
        win = ui.Window('Graphs')
        self.expr = ui.Entry('', 50, self.plot)
        win.add([self.expr, ui.helpbutton(graph_help_text)])
        win.add([ui.Button(_('Plot'), self.plot, 'xy'), ' x, y1, y2, ...'], 'w')
        win.add([ui.Button(_('Plot'), self.plot, 'y'), ' y1, y2, ...'], 'w')
        win.add([ui.Button(_('Save'), self.save)], 'w')
        self.gui = gui

    def plot(self, type=None, expr=None, ignore_if_nan=False):
        if expr is None:
            expr = self.expr.value
        else:
            self.expr.value = expr
        try:
            data = self.gui.images.graph(expr)
        except Exception as ex:
            ui.error(ex)
            return
        if ignore_if_nan and len(data) == 2 and np.isnan(data[1]).all():
            return
        pickledata = (data, self.gui.frame, expr, type)
        self.gui.pipe('graph', pickledata)

    def save(self):
        dialog = ui.SaveFileDialog(self.gui.window.win, _('Save data to file ... '))
        filename = dialog.go()
        if filename:
            expr = self.expr.value
            data = self.gui.images.graph(expr)
            np.savetxt(filename, data.T, header=expr)