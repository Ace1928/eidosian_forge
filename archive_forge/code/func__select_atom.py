from ase import Atoms
def _select_atom(self, chg=None):
    sel = self.asel.value
    self.view.remove_spacefill()
    for e in set(self.struct.get_chemical_symbols()):
        if sel == 'All' or e == sel:
            if e in self.colors:
                self.view.add_spacefill(selection='#' + e, color=self.colors[e])
            else:
                self.view.add_spacefill(selection='#' + e)
    self._update_repr()