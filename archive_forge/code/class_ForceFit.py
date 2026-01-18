import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
class ForceFit(namedtuple('ForceFit', ['path', 'energies', 'fit_path', 'fit_energies', 'lines'])):
    """Data container to hold fitting parameters for force curves."""

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.plot(self.path, self.energies, 'o')
        for x, y in self.lines:
            ax.plot(x, y, '-g')
        ax.plot(self.fit_path, self.fit_energies, 'k-')
        ax.set_xlabel('path [Å]')
        ax.set_ylabel('energy [eV]')
        Ef = max(self.energies) - self.energies[0]
        Er = max(self.energies) - self.energies[-1]
        dE = self.energies[-1] - self.energies[0]
        ax.set_title('$E_\\mathrm{{f}} \\approx$ {:.3f} eV; $E_\\mathrm{{r}} \\approx$ {:.3f} eV; $\\Delta E$ = {:.3f} eV'.format(Ef, Er, dE))
        return ax