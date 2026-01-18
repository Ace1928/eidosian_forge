from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
class BoltztrapRunner(MSONable):
    """This class is used to run Boltztrap on a band structure object."""

    @requires(which('x_trans'), "BoltztrapRunner requires the executables 'x_trans' to be in PATH. Please download Boltztrap at http://www.icams.de/content/research/software-development/boltztrap/ and follow the instructions in the README to compile Bolztrap accordingly. Then add x_trans to your path")
    def __init__(self, bs, nelec, dos_type='HISTO', energy_grid=0.005, lpfac=10, run_type='BOLTZ', band_nb=None, tauref=0, tauexp=0, tauen=0, soc=False, doping=None, energy_span_around_fermi=1.5, scissor=0.0, kpt_line=None, spin=None, cond_band=False, tmax=1300, tgrid=50, symprec=0.001, cb_cut=10, timeout=7200) -> None:
        """
        Args:
            bs:
                A band structure object
            nelec:
                the number of electrons
            dos_type:
                two options for the band structure integration: "HISTO"
                (histogram) or "TETRA" using the tetrahedon method. TETRA
                typically gives better results (especially for DOSes)
                but takes more time
            energy_grid:
                the energy steps used for the integration (eV)
            lpfac:
                the number of interpolation points in the real space. By
                default 10 gives 10 time more points in the real space than
                the number of kpoints given in reciprocal space
            run_type:
                type of boltztrap usage. by default
                - BOLTZ: (default) compute transport coefficients
                - BANDS: interpolate all bands contained in the energy range
                specified in energy_span_around_fermi variable, along specified
                k-points
                - DOS: compute total and partial dos (custom BoltzTraP code
                needed!)
                - FERMI: compute fermi surface or more correctly to
                get certain bands interpolated
            band_nb:
                indicates a band number. Used for Fermi Surface interpolation
                (run_type="FERMI")
            spin:
                specific spin component (1: up, -1: down) of the band selected
                in FERMI mode (mandatory).
            cond_band:
                if a conduction band is specified in FERMI mode,
                set this variable as True
            tauref:
                reference relaxation time. Only set to a value different than
                zero if we want to model beyond the constant relaxation time.
            tauexp:
                exponent for the energy in the non-constant relaxation time
                approach
            tauen:
                reference energy for the non-constant relaxation time approach
            soc:
                results from spin-orbit coupling (soc) computations give
                typically non-polarized (no spin up or down) results but single
                electron occupations. If the band structure comes from a soc
                computation, you should set soc to True (default False)
            doping:
                the fixed doping levels you want to compute. Boltztrap provides
                both transport values depending on electron chemical potential
                (fermi energy) and for a series of fixed carrier
                concentrations. By default, this is set to 1e16 to 1e22 in
                increments of factors of 10.
            energy_span_around_fermi:
                usually the interpolation is not needed on the entire energy
                range but on a specific range around the Fermi level.
                This energy gives this range in eV. by default it is 1.5 eV.
                If DOS or BANDS type are selected, this range is automatically
                set to cover the entire energy range.
            scissor:
                scissor to apply to the band gap (eV). This applies a scissor
                operation moving the band edges without changing the band
                shape. This is useful to correct the often underestimated band
                gap in DFT. Default is 0.0 (no scissor)
            kpt_line:
                list of fractional coordinates of kpoints as arrays or list of
                Kpoint objects for BANDS mode calculation (standard path of
                high symmetry k-points is automatically set as default)
            tmax:
                Maximum temperature (K) for calculation (default=1300)
            tgrid:
                Temperature interval for calculation (default=50)
            symprec: 1e-3 is the default in pymatgen. If the kmesh has been
                generated using a different symprec, it has to be specified
                to avoid a "factorization error" in BoltzTraP calculation.
                If a kmesh that spans the whole Brillouin zone has been used,
                or to disable all the symmetries, set symprec to None.
            cb_cut: by default 10% of the highest conduction bands are
                removed because they are often not accurate.
                Tune cb_cut to change the percentage (0-100) of bands
                that are removed.
            timeout: overall time limit (in seconds): mainly to avoid infinite
                loop when trying to find Fermi levels.
        """
        self.lpfac = lpfac
        self._bs = bs
        self._nelec = nelec
        self.dos_type = dos_type
        self.energy_grid = energy_grid
        self.error: list[str] = []
        self.run_type = run_type
        self.band_nb = band_nb
        self.spin = spin
        self.cond_band = cond_band
        self.tauref = tauref
        self.tauexp = tauexp
        self.tauen = tauen
        self.soc = soc
        self.kpt_line = kpt_line
        self.cb_cut = cb_cut / 100.0
        if isinstance(doping, list) and len(doping) > 0:
            self.doping = doping
        else:
            self.doping = []
            for d in [1e+16, 1e+17, 1e+18, 1e+19, 1e+20, 1e+21]:
                self.doping.extend([1 * d, 2.5 * d, 5 * d, 7.5 * d])
            self.doping.append(1e+22)
        self.energy_span_around_fermi = energy_span_around_fermi
        self.scissor = scissor
        self.tmax = tmax
        self.tgrid = tgrid
        self._symprec = symprec
        if self.run_type in ('DOS', 'BANDS'):
            self._auto_set_energy_range()
        self.timeout = timeout
        self.start_time = time.perf_counter()

    def _auto_set_energy_range(self) -> None:
        """Automatically determine the energy range as min/max eigenvalue
        minus/plus the buffer_in_ev.
        """
        emins = [min((e_k[0] for e_k in self._bs.bands[Spin.up]))]
        emaxs = [max((e_k[0] for e_k in self._bs.bands[Spin.up]))]
        if self._bs.is_spin_polarized:
            emins.append(min((e_k[0] for e_k in self._bs.bands[Spin.down])))
            emaxs.append(max((e_k[0] for e_k in self._bs.bands[Spin.down])))
        min_eigenval = Energy(min(emins) - self._bs.efermi, 'eV').to('Ry')
        max_eigenval = Energy(max(emaxs) - self._bs.efermi, 'eV').to('Ry')
        const = Energy(2, 'eV').to('Ry')
        self._ll = min_eigenval - const
        self._hl = max_eigenval + const
        en_range = Energy(max((abs(self._ll), abs(self._hl))), 'Ry').to('eV')
        self.energy_span_around_fermi = en_range * 1.01
        print('energy_span_around_fermi = ', self.energy_span_around_fermi)

    @property
    def bs(self):
        """The BandStructure."""
        return self._bs

    @property
    def nelec(self):
        """Number of electrons."""
        return self._nelec

    def write_energy(self, output_file) -> None:
        """Writes the energy to an output file.

        Args:
            output_file: Filename
        """
        with open(output_file, mode='w') as file:
            file.write('test\n')
            file.write(f'{len(self._bs.kpoints)}\n')
            if self.run_type == 'FERMI':
                sign = -1.0 if self.cond_band else 1.0
                for i, kpt in enumerate(self._bs.kpoints):
                    eigs = []
                    eigs.append(Energy(self._bs.bands[Spin(self.spin)][self.band_nb][i] - self._bs.efermi, 'eV').to('Ry'))
                    a, b, c = kpt.frac_coords
                    file.write(f'{a:12.8f} {b:12.8f} {c:12.8f}{len(eigs)}\n')
                    for e in eigs:
                        file.write(f'{sign * float(e):18.8f}\n')
            else:
                for i, kpt in enumerate(self._bs.kpoints):
                    eigs = []
                    spin_lst = [self.spin] if self.run_type == 'DOS' else self._bs.bands
                    for spin in spin_lst:
                        nb_bands = int(math.floor(self._bs.nb_bands * (1 - self.cb_cut)))
                        for j in range(nb_bands):
                            eigs.append(Energy(self._bs.bands[Spin(spin)][j][i] - self._bs.efermi, 'eV').to('Ry'))
                    eigs.sort()
                    if self.run_type == 'DOS' and self._bs.is_spin_polarized:
                        eigs.insert(0, self._ll)
                        eigs.append(self._hl)
                    a, b, c = kpt.frac_coords
                    file.write(f'{a:12.8f} {b:12.8f} {c:12.8f} {len(eigs)}\n')
                    for e in eigs:
                        file.write(f'{float(e):18.8f}\n')

    def write_struct(self, output_file) -> None:
        """Writes the structure to an output file.

        Args:
            output_file: Filename
        """
        if self._symprec is not None:
            sym = SpacegroupAnalyzer(self._bs.structure, symprec=self._symprec)
        elif self._symprec is None:
            pass
        with open(output_file, mode='w') as file:
            if self._symprec is not None:
                file.write(f'{self._bs.structure.formula} {sym.get_space_group_symbol()}\n')
            elif self._symprec is None:
                file.write(f'{self._bs.structure.formula} symmetries disabled\n')
            file.write('\n'.join((' '.join((f'{Length(i, 'ang').to('bohr'):.5f}' for i in row)) for row in self._bs.structure.lattice.matrix)) + '\n')
            if self._symprec is not None:
                ops = sym.get_symmetry_dataset()['rotations']
            elif self._symprec is None:
                ops = [np.eye(3)]
            file.write(f'{len(ops)}\n')
            for op in ops:
                for row in op:
                    file.write(f'{' '.join(map(str, row))}\n')

    def write_def(self, output_file) -> None:
        """Writes the def to an output file.

        Args:
            output_file: Filename
        """
        with open(output_file, mode='w') as file:
            so = ''
            if self._bs.is_spin_polarized or self.soc:
                so = 'so'
            file.write(f"5, 'boltztrap.intrans',      'old',    'formatted',0\n6,'boltztrap.outputtrans',      'unknown',    'formatted',0\n20,'boltztrap.struct',         'old',    'formatted',0\n10,'boltztrap.energy{so}',         'old',    'formatted',0\n48,'boltztrap.engre',         'unknown',    'unformatted',0\n49,'boltztrap.transdos',        'unknown',    'formatted',0\n50,'boltztrap.sigxx',        'unknown',    'formatted',0\n51,'boltztrap.sigxxx',        'unknown',    'formatted',0\n21,'boltztrap.trace',           'unknown',    'formatted',0\n22,'boltztrap.condtens',           'unknown',    'formatted',0\n24,'boltztrap.halltens',           'unknown',    'formatted',0\n30,'boltztrap_BZ.cube',           'unknown',    'formatted',0\n")

    def write_proj(self, output_file_proj: str, output_file_def: str) -> None:
        """Writes the projections to an output file.

        Args:
            output_file_proj: output file name
            output_file_def: output file name
        """
        for oi, o in enumerate(Orbital):
            for site_nb in range(len(self._bs.structure)):
                if oi < len(self._bs.projections[Spin.up][0][0]):
                    with open(f'{output_file_proj}_{site_nb}_{o}', mode='w') as file:
                        file.write(self._bs.structure.formula + '\n')
                        file.write(str(len(self._bs.kpoints)) + '\n')
                        for i, kpt in enumerate(self._bs.kpoints):
                            tmp_proj = []
                            for j in range(int(math.floor(self._bs.nb_bands * (1 - self.cb_cut)))):
                                tmp_proj.append(self._bs.projections[Spin(self.spin)][j][i][oi][site_nb])
                            if self.run_type == 'DOS' and self._bs.is_spin_polarized:
                                tmp_proj.insert(0, self._ll)
                                tmp_proj.append(self._hl)
                            a, b, c = kpt.frac_coords
                            file.write(f'{a:12.8f} {b:12.8f} {c:12.8f} {len(tmp_proj)}\n')
                            for t in tmp_proj:
                                file.write(f'{float(t):18.8f}\n')
        with open(output_file_def, mode='w') as file:
            so = ''
            if self._bs.is_spin_polarized:
                so = 'so'
            file.write(f"5, 'boltztrap.intrans',      'old',    'formatted',0\n6,'boltztrap.outputtrans',      'unknown',    'formatted',0\n20,'boltztrap.struct',         'old',    'formatted',0\n10,'boltztrap.energy{so}',         'old',    'formatted',0\n48,'boltztrap.engre',         'unknown',    'unformatted',0\n49,'boltztrap.transdos',        'unknown',    'formatted',0\n50,'boltztrap.sigxx',        'unknown',    'formatted',0\n51,'boltztrap.sigxxx',        'unknown',    'formatted',0\n21,'boltztrap.trace',           'unknown',    'formatted',0\n22,'boltztrap.condtens',           'unknown',    'formatted',0\n24,'boltztrap.halltens',           'unknown',    'formatted',0\n30,'boltztrap_BZ.cube',           'unknown',    'formatted',0\n")
            i = 1000
            for oi, o in enumerate(Orbital):
                for site_nb in range(len(self._bs.structure)):
                    if oi < len(self._bs.projections[Spin.up][0][0]):
                        file.write(f"{i},'boltztrap.proj_{site_nb}_{o.name}old', 'formatted',0\n")
                        i += 1

    def write_intrans(self, output_file) -> None:
        """Writes the intrans to an output file.

        Args:
            output_file: Filename
        """
        setgap = 1 if self.scissor > 0.0001 else 0
        if self.run_type in ('BOLTZ', 'DOS'):
            with open(output_file, mode='w') as fout:
                fout.write('GENE          # use generic interface\n')
                fout.write(f'1 0 {setgap} {Energy(self.scissor, 'eV').to('Ry')}         # iskip (not presently used) idebug setgap shiftgap \n')
                fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} {Energy(self.energy_span_around_fermi, 'eV').to('Ry')} {self._nelec}.1f     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
                fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
                fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
                fout.write(f'{self.run_type}                     # run mode (only BOLTZ is supported)\n')
                fout.write('.15                       # (efcut) energy range of chemical potential\n')
                fout.write(f'{self.tmax} {self.tgrid}                  # Tmax, temperature grid\n')
                fout.write('-1.  # energyrange of bands given DOS output sig_xxx and dos_xxx (xxx is band number)\n')
                fout.write(self.dos_type + '\n')
                fout.write(f'{self.tauref} {self.tauexp} {self.tauen} 0 0 0\n')
                fout.write(f'{2 * len(self.doping)}\n')
                for d in self.doping:
                    fout.write(str(d) + '\n')
                for d in self.doping:
                    fout.write(str(-d) + '\n')
        elif self.run_type == 'FERMI':
            with open(output_file, mode='w') as fout:
                fout.write('GENE          # use generic interface\n')
                fout.write('1 0 0 0.0         # iskip (not presently used) idebug setgap shiftgap \n')
                fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} 0.1 {self._nelec:6.1f}     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
                fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
                fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
                fout.write('FERMI                     # run mode (only BOLTZ is supported)\n')
                fout.write(f'1                        # actual band selected: {self.band_nb + 1} spin: {self.spin}')
        elif self.run_type == 'BANDS':
            if self.kpt_line is None:
                kpath = HighSymmKpath(self._bs.structure)
                self.kpt_line = [Kpoint(k, self._bs.structure.lattice) for k in kpath.get_kpoints(coords_are_cartesian=False)[0]]
                self.kpt_line = [kp.frac_coords for kp in self.kpt_line]
            elif isinstance(self.kpt_line[0], Kpoint):
                self.kpt_line = [kp.frac_coords for kp in self.kpt_line]
            with open(output_file, mode='w') as fout:
                fout.write('GENE          # use generic interface\n')
                fout.write(f'1 0 {setgap} {Energy(self.scissor, 'eV').to('Ry')}         # iskip (not presently used) idebug setgap shiftgap \n')
                fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} {Energy(self.energy_span_around_fermi, 'eV').to('Ry')} {self._nelec:.1f}     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
                fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
                fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
                fout.write('BANDS                     # run mode (only BOLTZ is supported)\n')
                fout.write(f'P {len(self.kpt_line)}\n')
                for kp in self.kpt_line:
                    fout.writelines([f'{k} ' for k in kp])
                    fout.write('\n')

    def write_input(self, output_dir) -> None:
        """Writes the input files.

        Args:
            output_dir: Directory to write the input files.
        """
        if self._bs.is_spin_polarized or self.soc:
            self.write_energy(f'{output_dir}/boltztrap.energyso')
        else:
            self.write_energy(f'{output_dir}/boltztrap.energy')
        self.write_struct(f'{output_dir}/boltztrap.struct')
        self.write_intrans(f'{output_dir}/boltztrap.intrans')
        self.write_def(f'{output_dir}/BoltzTraP.def')
        if len(self.bs.projections) != 0 and self.run_type == 'DOS':
            self.write_proj(f'{output_dir}/boltztrap.proj', f'{output_dir}/BoltzTraP.def')

    def run(self, path_dir=None, convergence=True, write_input=True, clear_dir=False, max_lpfac=150, min_egrid=5e-05):
        """Write inputs (optional), run BoltzTraP, and ensure
        convergence (optional).

        Args:
            path_dir (str): directory in which to run BoltzTraP
            convergence (bool): whether to check convergence and make
                corrections if needed
            write_input: (bool) whether to write input files before the run
                (required for convergence mode)
            clear_dir: (bool) whether to remove all files in the path_dir
                before starting
            max_lpfac: (float) maximum lpfac value to try before reducing egrid
                in convergence mode
            min_egrid: (float) minimum egrid value to try before giving up in
                convergence mode
        """
        if convergence and (not write_input):
            raise ValueError('Convergence mode requires write_input to be true')
        run_type = self.run_type
        if run_type in ('BANDS', 'DOS', 'FERMI'):
            convergence = False
            if self.lpfac > max_lpfac:
                max_lpfac = self.lpfac
        if run_type == 'BANDS' and self.bs.is_spin_polarized:
            print(f'Reminder: for run_type={run_type!r}, spin component are not separated! (you have a spin polarized band structure)')
        if run_type in ('FERMI', 'DOS') and self.spin is None:
            if self.bs.is_spin_polarized:
                raise BoltztrapError('Spin parameter must be specified for spin polarized band structures!')
            self.spin = 1
        dir_bz_name = 'boltztrap'
        if path_dir is None:
            temp_dir = tempfile.mkdtemp()
            path_dir = os.path.join(temp_dir, dir_bz_name)
        else:
            path_dir = os.path.abspath(os.path.join(path_dir, dir_bz_name))
        os.mkdir(path_dir, exist_ok=True)
        if clear_dir:
            for c in os.listdir(path_dir):
                os.remove(os.path.join(path_dir, c))
        FORMAT = '%(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=f'{path_dir}/../boltztrap.out')
        with cd(path_dir):
            lpfac_start = self.lpfac
            converged = False
            while self.energy_grid >= min_egrid and (not converged):
                self.lpfac = lpfac_start
                if time.perf_counter() - self.start_time > self.timeout:
                    raise BoltztrapError(f'no doping convergence after timeout of {self.timeout} s')
                logging.info(f'lpfac, energy_grid: {self.lpfac} {self.energy_grid}')
                while self.lpfac <= max_lpfac and (not converged):
                    if time.perf_counter() - self.start_time > self.timeout:
                        raise BoltztrapError(f'no doping convergence after timeout of {self.timeout} s')
                    if write_input:
                        self.write_input(path_dir)
                    bt_exe = ['x_trans', 'BoltzTraP']
                    if self._bs.is_spin_polarized or self.soc:
                        bt_exe.append('-so')
                    with subprocess.Popen(bt_exe, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as p:
                        p.wait()
                        for c in p.communicate():
                            logging.info(c.decode())
                            if 'error in factorization' in c.decode():
                                raise BoltztrapError('error in factorization')
                    warning = ''
                    with open(os.path.join(path_dir, dir_bz_name + '.outputtrans')) as file:
                        for line in file:
                            if 'Option unknown' in line:
                                raise BoltztrapError('DOS mode needs a custom version of BoltzTraP code is needed')
                            if 'WARNING' in line:
                                warning = line
                                break
                            if 'Error - Fermi level was not found' in line:
                                warning = line
                                break
                    if not warning and convergence:
                        analyzer = BoltztrapAnalyzer.from_files(path_dir)
                        for doping in ['n', 'p']:
                            for c in analyzer.mu_doping[doping]:
                                if len(analyzer.mu_doping[doping][c]) != len(analyzer.doping[doping]):
                                    warning = 'length of mu_doping array is incorrect'
                                    break
                                if doping == 'p' and sorted(analyzer.mu_doping[doping][c], reverse=True) != analyzer.mu_doping[doping][c]:
                                    warning = 'sorting of mu_doping array incorrect for p-type'
                                    break
                                if doping == 'n' and sorted(analyzer.mu_doping[doping][c]) != analyzer.mu_doping[doping][c]:
                                    warning = 'sorting of mu_doping array incorrect for n-type'
                                    break
                    if warning:
                        self.lpfac += 10
                        logging.warning(f'Warning detected: {warning}! Increase lpfac to {self.lpfac}')
                    else:
                        converged = True
                if not converged:
                    self.energy_grid /= 10
                    logging.info(f'Could not converge with max lpfac; Decrease egrid to {self.energy_grid}')
            if not converged:
                lpfac, energy_grid = (self.lpfac, self.energy_grid)
                raise BoltztrapError(f'Doping convergence not reached with lpfac={lpfac!r}, energy_grid={energy_grid!r}')
            return path_dir

    def as_dict(self):
        """MSONable dict."""
        results = {'@module': type(self).__module__, '@class': type(self).__name__, 'lpfac': self.lpfac, 'bs': self.bs.as_dict(), 'nelec': self._nelec, 'dos_type': self.dos_type, 'run_type': self.run_type, 'band_nb': self.band_nb, 'spin': self.spin, 'cond_band': self.cond_band, 'tauref': self.tauref, 'tauexp': self.tauexp, 'tauen': self.tauen, 'soc': self.soc, 'kpt_line': self.kpt_line, 'doping': self.doping, 'energy_span_around_fermi': self.energy_span_around_fermi, 'scissor': self.scissor, 'tmax': self.tmax, 'tgrid': self.tgrid, 'symprec': self._symprec}
        return jsanitize(results)