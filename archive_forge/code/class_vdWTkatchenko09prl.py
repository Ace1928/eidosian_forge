import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list
from ase.parallel import world
from ase.utils import IOContext
class vdWTkatchenko09prl(Calculator, IOContext):
    """vdW correction after Tkatchenko and Scheffler PRL 102 (2009) 073005."""

    def __init__(self, hirshfeld=None, vdwradii=None, calculator=None, Rmax=10.0, Ldecay=1.0, vdWDB_alphaC6=vdWDB_alphaC6, txt=None, sR=None):
        """Constructor

        Parameters
        ==========
        hirshfeld: the Hirshfeld partitioning object
        calculator: the calculator to get the PBE energy
        """
        self.hirshfeld = hirshfeld
        if calculator is None:
            self.calculator = self.hirshfeld.get_calculator()
        else:
            self.calculator = calculator
        if txt is None:
            txt = get_logging_file_descriptor(self.calculator)
        if hasattr(self.calculator, 'world'):
            myworld = self.calculator.world
        else:
            myworld = world
        self.txt = self.openfile(txt, myworld)
        self.vdwradii = vdwradii
        self.vdWDB_alphaC6 = vdWDB_alphaC6
        self.Rmax = Rmax
        self.Ldecay = Ldecay
        self.atoms = None
        if sR is None:
            try:
                xc_name = self.calculator.get_xc_functional()
                self.sR = sR_opt[xc_name]
            except KeyError:
                raise ValueError('Tkatchenko-Scheffler dispersion correction not ' + 'implemented for %s functional' % xc_name)
        else:
            self.sR = sR
        self.d = 20
        Calculator.__init__(self)
        self.parameters['calculator'] = self.calculator.name
        self.parameters['xc'] = self.calculator.get_xc_functional()

    @property
    def implemented_properties(self):
        return self.calculator.implemented_properties

    def calculation_required(self, atoms, quantities):
        if self.calculator.calculation_required(atoms, quantities):
            return True
        for quantity in quantities:
            if quantity not in self.results:
                return True
        return False

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.update(atoms, properties)

    def update(self, atoms=None, properties=['energy', 'forces']):
        if not self.calculation_required(atoms, properties):
            return
        if atoms is None:
            atoms = self.calculator.get_atoms()
        properties = list(properties)
        for name in ('energy', 'forces'):
            if name not in properties:
                properties.append(name)
        for name in properties:
            self.results[name] = self.calculator.get_property(name, atoms)
        self.parameters['uncorrected_energy'] = self.results['energy']
        self.atoms = atoms.copy()
        if self.vdwradii is not None:
            vdwradii = self.vdwradii
            assert len(atoms) == len(vdwradii)
        else:
            vdwradii = []
            for atom in atoms:
                self.vdwradii.append(vdWDB_Grimme06jcc[atom.symbol][1])
        if self.hirshfeld is None:
            volume_ratios = [1.0] * len(atoms)
        elif hasattr(self.hirshfeld, '__len__'):
            assert len(atoms) == len(self.hirshfeld)
            volume_ratios = self.hirshfeld
        else:
            self.hirshfeld.initialize()
            volume_ratios = self.hirshfeld.get_effective_volume_ratios()
        na = len(atoms)
        C6eff_a = np.empty(na)
        alpha_a = np.empty(na)
        R0eff_a = np.empty(na)
        for a, atom in enumerate(atoms):
            alpha_a[a], C6eff_a[a] = self.vdWDB_alphaC6[atom.symbol]
            C6eff_a[a] *= Hartree * volume_ratios[a] ** 2 * Bohr ** 6
            R0eff_a[a] = vdwradii[a] * volume_ratios[a] ** (1 / 3.0)
        C6eff_aa = np.empty((na, na))
        for a in range(na):
            for b in range(a, na):
                C6eff_aa[a, b] = 2 * C6eff_a[a] * C6eff_a[b] / (alpha_a[b] / alpha_a[a] * C6eff_a[a] + alpha_a[a] / alpha_a[b] * C6eff_a[b])
                C6eff_aa[b, a] = C6eff_aa[a, b]
        pbc_c = atoms.get_pbc()
        EvdW = 0.0
        forces = 0.0 * self.results['forces']
        if pbc_c.any():
            tol = 1e-05
            Reff = self.Rmax + self.Ldecay * erfinv(1.0 - 2.0 * tol)
            n_list = neighbor_list(quantities='ijdDS', a=atoms, cutoff=Reff, self_interaction=False)
            atom_list = [[] for _ in range(0, len(atoms))]
            d_list = [[] for _ in range(0, len(atoms))]
            v_list = [[] for _ in range(0, len(atoms))]
            for k in range(0, len(n_list[0])):
                i = n_list[0][k]
                j = n_list[1][k]
                dist = n_list[2][k]
                vect = n_list[3][k]
                if j >= i:
                    atom_list[i].append(j)
                    d_list[i].append(dist)
                    v_list[i].append(vect)
        else:
            atom_list = []
            d_list = []
            v_list = []
            for i in range(0, len(atoms)):
                atom_list.append(range(i + 1, len(atoms)))
                d_list.append([atoms.get_distance(i, j) for j in range(i + 1, len(atoms))])
                v_list.append([atoms.get_distance(i, j, vector=True) for j in range(i + 1, len(atoms))])
        for i in range(len(atoms)):
            for j, r, vect in zip(atom_list[i], d_list[i], v_list[i]):
                r6 = r ** 6
                Edamp, Fdamp = self.damping(r, R0eff_a[i], R0eff_a[j], d=self.d, sR=self.sR)
                if pbc_c.any():
                    smooth = 0.5 * erfc((r - self.Rmax) / self.Ldecay)
                    smooth_der = -1.0 / np.sqrt(np.pi) / self.Ldecay * np.exp(-((r - self.Rmax) / self.Ldecay) ** 2)
                else:
                    smooth = 1.0
                    smooth_der = 0.0
                if i == j:
                    EvdW -= Edamp * C6eff_aa[i, j] / r6 / 2.0 * smooth
                else:
                    EvdW -= Edamp * C6eff_aa[i, j] / r6 * smooth
                if i != j:
                    force_ij = -((Fdamp - 6 * Edamp / r) * C6eff_aa[i, j] / r6 * smooth + Edamp * C6eff_aa[i, j] / r6 * smooth_der) * vect / r
                    forces[i] += force_ij
                    forces[j] -= force_ij
        self.results['energy'] += EvdW
        self.results['forces'] += forces
        if self.txt:
            print('\n' + self.__class__.__name__, file=self.txt)
            print('vdW correction: %g' % EvdW, file=self.txt)
            print('Energy:         %g' % self.results['energy'], file=self.txt)
            print('\nForces in eV/Ang:', file=self.txt)
            symbols = self.atoms.get_chemical_symbols()
            for ia, symbol in enumerate(symbols):
                print('%3d %-2s %10.5f %10.5f %10.5f' % ((ia, symbol) + tuple(self.results['forces'][ia])), file=self.txt)
            self.txt.flush()

    def damping(self, RAB, R0A, R0B, d=20, sR=0.94):
        """Damping factor.

        Standard values for d and sR as given in
        Tkatchenko and Scheffler PRL 102 (2009) 073005."""
        scale = 1.0 / (sR * (R0A + R0B))
        x = RAB * scale
        chi = np.exp(-d * (x - 1.0))
        return (1.0 / (1.0 + chi), d * scale * chi / (1.0 + chi) ** 2)