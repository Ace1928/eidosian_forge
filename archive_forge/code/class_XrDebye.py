from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
class XrDebye:
    """
    Class for calculation of XRD or SAXS patterns.
    """

    def __init__(self, atoms, wavelength, damping=0.04, method='Iwasa', alpha=1.01, warn=True):
        """
        Initilize the calculation of X-ray diffraction patterns

        Parameters:

        atoms: ase.Atoms
            atoms object for which calculation will be performed.

        wavelength: float, Angstrom
            X-ray wavelength in Angstrom. Used for XRD and to setup dumpings.

        damping : float, Angstrom**2
            thermal damping factor parameter (B-factor).

        method: {'Iwasa'}
            method of calculation (damping and atomic factors affected).

            If set to 'Iwasa' than angular damping and q-dependence of
            atomic factors are used.

            For any other string there will be only thermal damping
            and constant atomic factors (`f_a(q) = Z_a`).

        alpha: float
            parameter for angular damping of scattering intensity.
            Close to 1.0 for unplorized beam.

        warn: boolean
            flag to show warning if atomic factor can't be calculated
        """
        self.wavelength = wavelength
        self.damping = damping
        self.mode = ''
        self.method = method
        self.alpha = alpha
        self.warn = warn
        self.twotheta_list = []
        self.q_list = []
        self.intensity_list = []
        self.atoms = atoms

    def set_damping(self, damping):
        """ set B-factor for thermal damping """
        self.damping = damping

    def get(self, s):
        """Get the powder x-ray (XRD) scattering intensity
        using the Debye-Formula at single point.

        Parameters:

        s: float, in inverse Angstrom
            scattering vector value (`s = q / 2\\pi`).

        Returns:
            Intensity at given scattering vector `s`.
        """
        pre = exp(-self.damping * s ** 2 / 2)
        if self.method == 'Iwasa':
            sinth = self.wavelength * s / 2.0
            positive = 1.0 - sinth ** 2
            if positive < 0:
                positive = 0
            costh = sqrt(positive)
            cos2th = cos(2.0 * acos(costh))
            pre *= costh / (1.0 + self.alpha * cos2th ** 2)
        f = {}

        def atomic(symbol):
            """
            get atomic factor, using cache.
            """
            if symbol not in f:
                if self.method == 'Iwasa':
                    f[symbol] = self.get_waasmaier(symbol, s)
                else:
                    f[symbol] = atomic_numbers[symbol]
            return f[symbol]
        I = 0.0
        fa = []
        for a in self.atoms:
            fa.append(atomic(a.symbol))
        pos = self.atoms.get_positions()
        fa = np.array(fa)
        for i in range(len(self.atoms)):
            vr = pos - pos[i]
            I += np.sum(fa[i] * fa * np.sinc(2 * s * np.sqrt(np.sum(vr * vr, axis=1))))
        return pre * I

    def get_waasmaier(self, symbol, s):
        """Scattering factor for free atoms.

        Parameters:

        symbol: string
            atom element symbol.

        s: float, in inverse Angstrom
            scattering vector value (`s = q / 2\\pi`).

        Returns:
            Intensity at given scattering vector `s`.

        Note:
            for hydrogen will be returned zero value."""
        if symbol == 'H':
            return 0
        elif symbol in waasmaier:
            abc = waasmaier[symbol]
            f = abc[10]
            s2 = s * s
            for i in range(5):
                f += abc[2 * i] * exp(-abc[2 * i + 1] * s2)
            return f
        if self.warn:
            print('<xrdebye::get_atomic> Element', symbol, 'not available')
        return 0

    def calc_pattern(self, x=None, mode='XRD', verbose=False):
        """
        Calculate X-ray diffraction pattern or
        small angle X-ray scattering pattern.

        Parameters:

        x: float array
            points where intensity will be calculated.
            XRD - 2theta values, in degrees;
            SAXS - q values in 1/A
            (`q = 2 \\pi \\cdot s = 4 \\pi \\sin( \\theta) / \\lambda`).
            If ``x`` is ``None`` then default values will be used.

        mode: {'XRD', 'SAXS'}
            the mode of calculation: X-ray diffraction (XRD) or
            small-angle scattering (SAXS).

        Returns:
            list of intensities calculated for values given in ``x``.
        """
        self.mode = mode.upper()
        assert mode in ['XRD', 'SAXS']
        result = []
        if mode == 'XRD':
            if x is None:
                self.twotheta_list = np.linspace(15, 55, 100)
            else:
                self.twotheta_list = x
            self.q_list = []
            if verbose:
                print('#2theta\tIntensity')
            for twotheta in self.twotheta_list:
                s = 2 * sin(twotheta * pi / 180 / 2.0) / self.wavelength
                result.append(self.get(s))
                if verbose:
                    print('%.3f\t%f' % (twotheta, result[-1]))
        elif mode == 'SAXS':
            if x is None:
                self.twotheta_list = np.logspace(-3, -0.3, 100)
            else:
                self.q_list = x
            self.twotheta_list = []
            if verbose:
                print('#q\tIntensity')
            for q in self.q_list:
                s = q / (2 * pi)
                result.append(self.get(s))
                if verbose:
                    print('%.4f\t%f' % (q, result[-1]))
        self.intensity_list = np.array(result)
        return self.intensity_list

    def write_pattern(self, filename):
        """ Save calculated data to file specified by ``filename`` string."""
        with open(filename, 'w') as fd:
            self._write_pattern(fd)

    def _write_pattern(self, fd):
        fd.write('# Wavelength = %f\n' % self.wavelength)
        if self.mode == 'XRD':
            x, y = (self.twotheta_list, self.intensity_list)
            fd.write('# 2theta \t Intesity\n')
        elif self.mode == 'SAXS':
            x, y = (self.q_list, self.intensity_list)
            fd.write('# q(1/A)\tIntesity\n')
        else:
            raise Exception('No data available, call calc_pattern() first.')
        for i in range(len(x)):
            fd.write('  %f\t%f\n' % (x[i], y[i]))

    def plot_pattern(self, filename=None, show=False, ax=None):
        """ Plot XRD or SAXS depending on filled data

        Uses Matplotlib to plot pattern. Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file.

        Returns:
            ``matplotlib.axes.Axes`` object."""
        import matplotlib.pyplot as plt
        if ax is None:
            plt.clf()
            ax = plt.gca()
        if self.mode == 'XRD':
            x, y = (np.array(self.twotheta_list), np.array(self.intensity_list))
            ax.plot(x, y / np.max(y), '.-')
            ax.set_xlabel('2$\\theta$')
            ax.set_ylabel('Intensity')
        elif self.mode == 'SAXS':
            x, y = (np.array(self.q_list), np.array(self.intensity_list))
            ax.loglog(x, y / np.max(y), '.-')
            ax.set_xlabel('q, 1/Angstr.')
            ax.set_ylabel('Intensity')
        else:
            raise Exception('No data available, call calc_pattern() first')
        if show:
            plt.show()
        if filename is not None:
            fig = ax.get_figure()
            fig.savefig(filename)
        return ax