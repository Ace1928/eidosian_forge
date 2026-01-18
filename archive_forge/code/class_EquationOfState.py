import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
class EquationOfState:
    """Fit equation of state for bulk systems.

    The following equation is used::

        sjeos (default)
            A third order inverse polynomial fit 10.1103/PhysRevB.67.026103

            ::

                                    2      3        -1/3
                E(V) = c + c t + c t  + c t ,  t = V
                        0   1     2      3

        taylor
            A third order Taylor series expansion about the minimum volume

        murnaghan
            PRB 28, 5480 (1983)

        birch
            Intermetallic compounds: Principles and Practice,
            Vol I: Principles. pages 195-210

        birchmurnaghan
            PRB 70, 224107

        pouriertarantola
            PRB 70, 224107

        vinet
            PRB 70, 224107

        antonschmidt
            Intermetallics 11, 23-32 (2003)

        p3
            A third order polynomial fit

    Use::

        eos = EquationOfState(volumes, energies, eos='murnaghan')
        v0, e0, B = eos.fit()
        eos.plot(show=True)

    """

    def __init__(self, volumes, energies, eos='sj'):
        self.v = np.array(volumes)
        self.e = np.array(energies)
        if eos == 'sjeos':
            eos = 'sj'
        self.eos_string = eos
        self.v0 = None

    def fit(self, warn=True):
        """Calculate volume, energy, and bulk modulus.

        Returns the optimal volume, the minimum energy, and the bulk
        modulus.  Notice that the ASE units for the bulk modulus is
        eV/Angstrom^3 - to get the value in GPa, do this::

          v0, e0, B = eos.fit()
          print(B / kJ * 1.0e24, 'GPa')

        """
        if self.eos_string == 'sj':
            return self.fit_sjeos()
        self.func = globals()[self.eos_string]
        p0 = [min(self.e), 1, 1]
        popt, pcov = curve_fit(parabola, self.v, self.e, p0)
        parabola_parameters = popt
        minvol = min(self.v)
        maxvol = max(self.v)
        c = parabola_parameters[2]
        b = parabola_parameters[1]
        a = parabola_parameters[0]
        parabola_vmin = -b / 2 / c
        E0 = parabola(parabola_vmin, a, b, c)
        B0 = 2 * c * parabola_vmin
        if self.eos_string == 'antonschmidt':
            BP = -2
        else:
            BP = 4
        initial_guess = [E0, B0, BP, parabola_vmin]
        p0 = initial_guess
        popt, pcov = curve_fit(self.func, self.v, self.e, p0)
        self.eos_parameters = popt
        if self.eos_string == 'p3':
            c0, c1, c2, c3 = self.eos_parameters
            a = 3 * c3
            b = 2 * c2
            c = c1
            self.v0 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            self.e0 = p3(self.v0, c0, c1, c2, c3)
            self.B = (2 * c2 + 6 * c3 * self.v0) * self.v0
        else:
            self.v0 = self.eos_parameters[3]
            self.e0 = self.eos_parameters[0]
            self.B = self.eos_parameters[1]
        if warn and (not minvol < self.v0 < maxvol):
            warnings.warn('The minimum volume of your fit is not in your volumes.  You may not have a minimum in your dataset!')
        return (self.v0, self.e0, self.B)

    def getplotdata(self):
        if self.v0 is None:
            self.fit()
        x = np.linspace(min(self.v), max(self.v), 100)
        if self.eos_string == 'sj':
            y = self.fit0(x ** (-(1 / 3)))
        else:
            y = self.func(x, *self.eos_parameters)
        return (self.eos_string, self.e0, self.v0, self.B, x, y, self.v, self.e)

    def plot(self, filename=None, show=False, ax=None):
        """Plot fitted energy curve.

        Uses Matplotlib to plot the energy curve.  Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file."""
        import matplotlib.pyplot as plt
        plotdata = self.getplotdata()
        ax = plot(*plotdata, ax=ax)
        if show:
            plt.show()
        if filename is not None:
            fig = ax.get_figure()
            fig.savefig(filename)
        return ax

    def fit_sjeos(self):
        """Calculate volume, energy, and bulk modulus.

        Returns the optimal volume, the minimum energy, and the bulk
        modulus.  Notice that the ASE units for the bulk modulus is
        eV/Angstrom^3 - to get the value in GPa, do this::

          v0, e0, B = eos.fit()
          print(B / kJ * 1.0e24, 'GPa')

        """
        fit0 = np.poly1d(np.polyfit(self.v ** (-(1 / 3)), self.e, 3))
        fit1 = np.polyder(fit0, 1)
        fit2 = np.polyder(fit1, 1)
        self.v0 = None
        for t in np.roots(fit1):
            if isinstance(t, float) and t > 0 and (fit2(t) > 0):
                self.v0 = t ** (-3)
                break
        if self.v0 is None:
            raise ValueError('No minimum!')
        self.e0 = fit0(t)
        self.B = t ** 5 * fit2(t) / 9
        self.fit0 = fit0
        return (self.v0, self.e0, self.B)