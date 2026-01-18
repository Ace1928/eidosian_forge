import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
class GenerateVaspInput:
    xc_defaults = {'lda': {'pp': 'LDA'}, 'pw91': {'pp': 'PW91', 'gga': '91'}, 'pbe': {'pp': 'PBE', 'gga': 'PE'}, 'pbesol': {'gga': 'PS'}, 'revpbe': {'gga': 'RE'}, 'rpbe': {'gga': 'RP'}, 'am05': {'gga': 'AM'}, 'tpss': {'metagga': 'TPSS'}, 'revtpss': {'metagga': 'RTPSS'}, 'm06l': {'metagga': 'M06L'}, 'ms0': {'metagga': 'MS0'}, 'ms1': {'metagga': 'MS1'}, 'ms2': {'metagga': 'MS2'}, 'scan': {'metagga': 'SCAN'}, 'scan-rvv10': {'metagga': 'SCAN', 'luse_vdw': True, 'bparam': 15.7}, 'mbj': {'metagga': 'MBJ'}, 'tb09': {'metagga': 'MBJ'}, 'vdw-df': {'gga': 'RE', 'luse_vdw': True, 'aggac': 0.0}, 'vdw-df-cx': {'gga': 'CX', 'luse_vdw': True, 'aggac': 0.0}, 'vdw-df-cx0p': {'gga': 'CX', 'luse_vdw': True, 'aggac': 0.0, 'lhfcalc': True, 'aexx': 0.2, 'aggax': 0.8}, 'optpbe-vdw': {'gga': 'OR', 'luse_vdw': True, 'aggac': 0.0}, 'optb88-vdw': {'gga': 'BO', 'luse_vdw': True, 'aggac': 0.0, 'param1': 1.1 / 6.0, 'param2': 0.22}, 'optb86b-vdw': {'gga': 'MK', 'luse_vdw': True, 'aggac': 0.0, 'param1': 0.1234, 'param2': 1.0}, 'vdw-df2': {'gga': 'ML', 'luse_vdw': True, 'aggac': 0.0, 'zab_vdw': -1.8867}, 'rev-vdw-df2': {'gga': 'MK', 'luse_vdw': True, 'param1': 0.1234, 'param2': 0.711357, 'zab_vdw': -1.8867, 'aggac': 0.0}, 'beef-vdw': {'gga': 'BF', 'luse_vdw': True, 'zab_vdw': -1.8867}, 'hf': {'lhfcalc': True, 'aexx': 1.0, 'aldac': 0.0, 'aggac': 0.0}, 'b3lyp': {'gga': 'B3', 'lhfcalc': True, 'aexx': 0.2, 'aggax': 0.72, 'aggac': 0.81, 'aldac': 0.19}, 'pbe0': {'gga': 'PE', 'lhfcalc': True}, 'hse03': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.3}, 'hse06': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.2}, 'hsesol': {'gga': 'PS', 'lhfcalc': True, 'hfscreen': 0.2}, 'sogga': {'gga': 'SA'}, 'sogga11': {'gga': 'S1'}, 'sogga11-x': {'gga': 'SX', 'lhfcalc': True, 'aexx': 0.401}, 'n12': {'gga': 'N2'}, 'n12-sx': {'gga': 'NX', 'lhfcalc': True, 'lhfscreen': 0.2}, 'mn12l': {'metagga': 'MN12L'}, 'gam': {'gga': 'GA'}, 'mn15l': {'metagga': 'MN15L'}, 'hle17': {'metagga': 'HLE17'}, 'revm06l': {'metagga': 'revM06L'}, 'm06sx': {'metagga': 'M06SX', 'lhfcalc': True, 'hfscreen': 0.189, 'aexx': 0.335}}
    VASP_PP_PATH = 'VASP_PP_PATH'

    def __init__(self, restart=None):
        self.float_params = {}
        self.exp_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_bool_params = {}
        self.list_int_params = {}
        self.list_float_params = {}
        self.special_params = {}
        self.dict_params = {}
        for key in float_keys:
            self.float_params[key] = None
        for key in exp_keys:
            self.exp_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_bool_keys:
            self.list_bool_params[key] = None
        for key in list_int_keys:
            self.list_int_params[key] = None
        for key in list_float_keys:
            self.list_float_params[key] = None
        for key in special_keys:
            self.special_params[key] = None
        for key in dict_keys:
            self.dict_params[key] = None
        self.input_params = {'xc': None, 'pp': None, 'setups': None, 'txt': '-', 'kpts': (1, 1, 1), 'gamma': False, 'kpts_nintersections': None, 'reciprocal': False, 'ignore_constraints': False, 'charge': None, 'net_charge': None, 'custom': {}}

    def set_xc_params(self, xc):
        """Set parameters corresponding to XC functional"""
        xc = xc.lower()
        if xc is None:
            pass
        elif xc not in self.xc_defaults:
            xc_allowed = ', '.join(self.xc_defaults.keys())
            raise ValueError('{0} is not supported for xc! Supported xc valuesare: {1}'.format(xc, xc_allowed))
        else:
            if 'pp' not in self.xc_defaults[xc]:
                self.set(pp='PBE')
            self.set(**self.xc_defaults[xc])

    def set(self, **kwargs):
        if 'ldauu' in kwargs and 'ldaul' in kwargs and ('ldauj' in kwargs) and ('ldau_luj' in kwargs):
            raise NotImplementedError("You can either specify ldaul, ldauu, and ldauj OR ldau_luj. ldau_luj is not a VASP keyword. It is a dictionary that specifies L, U and J for each chemical species in the atoms object. For example for a water molecule:ldau_luj={'H':{'L':2, 'U':4.0, 'J':0.9},\n                      'O':{'L':2, 'U':4.0, 'J':0.9}}")
        if 'xc' in kwargs:
            self.set_xc_params(kwargs['xc'])
        for key in kwargs:
            if key in self.float_params:
                self.float_params[key] = kwargs[key]
            elif key in self.exp_params:
                self.exp_params[key] = kwargs[key]
            elif key in self.string_params:
                self.string_params[key] = kwargs[key]
            elif key in self.int_params:
                self.int_params[key] = kwargs[key]
            elif key in self.bool_params:
                self.bool_params[key] = kwargs[key]
            elif key in self.list_bool_params:
                self.list_bool_params[key] = kwargs[key]
            elif key in self.list_int_params:
                self.list_int_params[key] = kwargs[key]
            elif key in self.list_float_params:
                self.list_float_params[key] = kwargs[key]
            elif key in self.special_params:
                self.special_params[key] = kwargs[key]
            elif key in self.dict_params:
                self.dict_params[key] = kwargs[key]
            elif key in self.input_params:
                self.input_params[key] = kwargs[key]
            else:
                raise TypeError('Parameter not defined: ' + key)

    def check_xc(self):
        """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """
        p = self.input_params
        if 'pp' not in p or p['pp'] is None:
            if self.string_params['gga'] is None:
                p.update({'pp': 'lda'})
            elif self.string_params['gga'] == '91':
                p.update({'pp': 'pw91'})
            elif self.string_params['gga'] == 'PE':
                p.update({'pp': 'pbe'})
            else:
                raise NotImplementedError("Unable to guess the desired set of pseudopotential(POTCAR) files. Please do one of the following: \n1. Use the 'xc' parameter to define your XC functional.These 'recipes' determine the pseudopotential file as well as setting the INCAR parameters.\n2. Use the 'gga' settings None (default), 'PE' or '91'; these correspond to LDA, PBE and PW91 respectively.\n3. Set the POTCAR explicitly with the 'pp' flag. The value should be the name of a folder on the VASP_PP_PATH, and the aliases 'LDA', 'PBE' and 'PW91' are alsoaccepted.\n")
        if p['xc'] is not None and p['xc'].lower() == 'lda' and (p['pp'].lower() != 'lda'):
            warnings.warn('XC is set to LDA, but PP is set to {0}. \nThis calculation is using the {0} POTCAR set. \n Please check that this is really what you intended!\n'.format(p['pp'].upper()))

    def _make_sort(self, atoms: ase.Atoms, special_setups: Sequence[int]=()) -> Tuple[List[int], List[int]]:
        symbols, _ = count_symbols(atoms, exclude=special_setups)
        srt = []
        srt.extend(special_setups)
        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if m in special_setups:
                    continue
                if atom.symbol == symbol:
                    srt.append(m)
        resrt = list(range(len(srt)))
        for n in range(len(resrt)):
            resrt[srt[n]] = n
        return (srt, resrt)

    def _build_pp_list(self, atoms, setups=None, special_setups: Sequence[int]=()):
        """Build the pseudopotential lists"""
        p = self.input_params
        if setups is None:
            setups, special_setups = self._get_setups()
        symbols, _ = count_symbols(atoms, exclude=special_setups)
        for pp_alias, pp_folder in (('lda', 'potpaw'), ('pw91', 'potpaw_GGA'), ('pbe', 'potpaw_PBE')):
            if p['pp'].lower() == pp_alias:
                break
        else:
            pp_folder = p['pp']
        if self.VASP_PP_PATH in os.environ:
            pppaths = os.environ[self.VASP_PP_PATH].split(':')
        else:
            pppaths = []
        ppp_list = []
        for m in special_setups:
            if m in setups:
                special_setup_index = m
            elif str(m) in setups:
                special_setup_index = str(m)
            else:
                raise Exception('Having trouble with special setup index {0}. Please use an int.'.format(m))
            potcar = join(pp_folder, setups[special_setup_index], 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)
                if isfile(filename) or islink(filename):
                    ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    ppp_list.append(filename + '.Z')
                    break
            else:
                symbol = atoms.symbols[m]
                msg = 'Looking for {}.\n                No pseudopotential for symbol{} with setup {} '.format(potcar, symbol, setups[special_setup_index])
                raise RuntimeError(msg)
        for symbol in symbols:
            try:
                potcar = join(pp_folder, symbol + setups[symbol], 'POTCAR')
            except (TypeError, KeyError):
                potcar = join(pp_folder, symbol, 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)
                if isfile(filename) or islink(filename):
                    ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    ppp_list.append(filename + '.Z')
                    break
            else:
                msg = 'Looking for PP for {}\n                        The pseudopotentials are expected to be in:\n                        LDA:  $VASP_PP_PATH/potpaw/\n                        PBE:  $VASP_PP_PATH/potpaw_PBE/\n                        PW91: $VASP_PP_PATH/potpaw_GGA/\n                        \n                        No pseudopotential for {}!'.format(potcar, symbol)
                raise RuntimeError(msg)
        return ppp_list

    def _get_setups(self):
        p = self.input_params
        special_setups = []
        setups_defaults = get_default_setups()
        if p['setups'] is None:
            p['setups'] = {'base': 'minimal'}
        elif isinstance(p['setups'], str):
            if p['setups'].lower() in setups_defaults.keys():
                p['setups'] = {'base': p['setups']}
        if 'base' in p['setups']:
            setups = setups_defaults[p['setups']['base'].lower()]
        else:
            setups = {}
        if p['setups'] is not None:
            setups.update(p['setups'])
        for m in setups:
            try:
                special_setups.append(int(m))
            except ValueError:
                pass
        return (setups, special_setups)

    def initialize(self, atoms):
        """Initialize a VASP calculation

        Constructs the POTCAR file (does not actually write it).
        User should specify the PATH
        to the pseudopotentials in VASP_PP_PATH environment variable

        The pseudopotentials are expected to be in:
        LDA:  $VASP_PP_PATH/potpaw/
        PBE:  $VASP_PP_PATH/potpaw_PBE/
        PW91: $VASP_PP_PATH/potpaw_GGA/

        if your pseudopotentials are somewhere else, or named
        differently you may make symlinks at the paths above that
        point to the right place. Alternatively, you may pass the full
        name of a folder on the VASP_PP_PATH to the 'pp' parameter.
        """
        self.check_xc()
        self.atoms = atoms
        self.all_symbols = atoms.get_chemical_symbols()
        self.natoms = len(atoms)
        self.spinpol = atoms.get_initial_magnetic_moments().any() or self.int_params['ispin'] == 2
        setups, special_setups = self._get_setups()
        symbols, symbolcount = count_symbols(atoms, exclude=special_setups)
        self.sort, self.resort = self._make_sort(atoms, special_setups=special_setups)
        self.atoms_sorted = atoms[self.sort]
        atomtypes = atoms.get_chemical_symbols()
        self.symbol_count = []
        for m in special_setups:
            self.symbol_count.append([atomtypes[m], 1])
        for m in symbols:
            self.symbol_count.append([m, symbolcount[m]])
        self.ppp_list = self._build_pp_list(atoms, setups=setups, special_setups=special_setups)
        self.converged = None
        self.setups_changed = None

    def default_nelect_from_ppp(self):
        """ Get default number of electrons from ppp_list and symbol_count

        "Default" here means that the resulting cell would be neutral.
        """
        symbol_valences = []
        for filename in self.ppp_list:
            with open_potcar(filename=filename) as ppp_file:
                r = read_potcar_numbers_of_electrons(ppp_file)
                symbol_valences.extend(r)
        assert len(self.symbol_count) == len(symbol_valences)
        default_nelect = 0
        for (symbol1, count), (symbol2, valence) in zip(self.symbol_count, symbol_valences):
            assert symbol1 == symbol2
            default_nelect += count * valence
        return default_nelect

    def write_input(self, atoms, directory='./'):
        from ase.io.vasp import write_vasp
        write_vasp(join(directory, 'POSCAR'), self.atoms_sorted, symbol_count=self.symbol_count, ignore_constraints=self.input_params['ignore_constraints'])
        self.write_incar(atoms, directory=directory)
        self.write_potcar(directory=directory)
        self.write_kpoints(atoms=atoms, directory=directory)
        self.write_sort_file(directory=directory)
        self.copy_vdw_kernel(directory=directory)

    def copy_vdw_kernel(self, directory='./'):
        """Method to copy the vdw_kernel.bindat file.
        Set ASE_VASP_VDW environment variable to the vdw_kernel.bindat
        folder location. Checks if LUSE_VDW is enabled, and if no location
        for the vdW kernel is specified, a warning is issued."""
        vdw_env = 'ASE_VASP_VDW'
        kernel = 'vdw_kernel.bindat'
        dst = os.path.join(directory, kernel)
        if isfile(dst):
            return
        if self.bool_params['luse_vdw']:
            src = None
            if vdw_env in os.environ:
                src = os.path.join(os.environ[vdw_env], kernel)
            if not src or not isfile(src):
                warnings.warn('vdW has been enabled, however no location for the {} file has been specified. Set {} environment variable to copy the vdW kernel.'.format(kernel, vdw_env))
            else:
                shutil.copyfile(src, dst)

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = ['CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT', 'POTCAR', 'vasprun.xml', 'WAVECAR', 'XDATCAR', 'PROCAR', 'ase-sort.dat', 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def write_incar(self, atoms, directory='./', **kwargs):
        """Writes the INCAR file."""
        p = self.input_params
        magmom_written = False
        incar = open(join(directory, 'INCAR'), 'w')
        incar.write('INCAR created by Atomic Simulation Environment\n')
        for key, val in self.float_params.items():
            if key == 'nelect':
                charge = p.get('charge')
                net_charge = p.get('net_charge')
                if net_charge is not None:
                    warnings.warn('`net_charge`, which is given in units of the *negative* elementary charge (i.e., the opposite of what one normally calls charge) has been deprecated in favor of `charge`, which is given in units of the positive elementary charge as usual', category=FutureWarning)
                    if charge is not None and charge != -net_charge:
                        raise ValueError("can't give both net_charge and charge")
                    charge = -net_charge
                if charge is not None and (charge != 0 or val is not None):
                    default_nelect = self.default_nelect_from_ppp()
                    nelect_from_charge = default_nelect - charge
                    if val is not None and val != nelect_from_charge:
                        raise ValueError('incompatible input parameters: nelect=%s, but charge=%s (neutral nelect is %s)' % (val, charge, default_nelect))
                    val = nelect_from_charge
            if val is not None:
                incar.write(' %s = %5.6f\n' % (key.upper(), val))
        for key, val in self.exp_params.items():
            if val is not None:
                incar.write(' %s = %5.2e\n' % (key.upper(), val))
        for key, val in self.string_params.items():
            if val is not None:
                incar.write(' %s = %s\n' % (key.upper(), val))
        for key, val in self.int_params.items():
            if val is not None:
                incar.write(' %s = %d\n' % (key.upper(), val))
                if key == 'ichain' and val > 0:
                    incar.write(' IBRION = 3\n POTIM = 0.0\n')
                    for key, val in self.int_params.items():
                        if key == 'iopt' and val is None:
                            print('WARNING: optimization is set to LFBGS (IOPT = 1)')
                            incar.write(' IOPT = 1\n')
                    for key, val in self.exp_params.items():
                        if key == 'ediffg' and val is None:
                            RuntimeError('Please set EDIFFG < 0')
        for key, val in self.list_bool_params.items():
            if val is None:
                pass
            else:
                incar.write(' %s = ' % key.upper())
                [incar.write('%s ' % _to_vasp_bool(x)) for x in val]
                incar.write('\n')
        for key, val in self.list_int_params.items():
            if val is None:
                pass
            elif key == 'ldaul' and self.dict_params['ldau_luj'] is not None:
                pass
            else:
                incar.write(' %s = ' % key.upper())
                [incar.write('%d ' % x) for x in val]
                incar.write('\n')
        for key, val in self.list_float_params.items():
            if val is None:
                pass
            elif key in ('ldauu', 'ldauj') and self.dict_params['ldau_luj'] is not None:
                pass
            elif key == 'magmom':
                if not len(val) == len(atoms):
                    msg = 'Expected length of magmom tag to be {}, i.e. 1 value per atom, but got {}'.format(len(atoms), len(val))
                    raise ValueError(msg)
                if not self.int_params['ispin']:
                    self.spinpol = True
                    incar.write(' ispin = 2\n'.upper())
                incar.write(' %s = ' % key.upper())
                magmom_written = True
                val = np.array(val)
                val = val[self.sort]
                lst = [[1, val[0]]]
                for n in range(1, len(val)):
                    if val[n] == val[n - 1]:
                        lst[-1][0] += 1
                    else:
                        lst.append([1, val[n]])
                incar.write(' '.join(['{:d}*{:.4f}'.format(mom[0], mom[1]) for mom in lst]))
                incar.write('\n')
            else:
                incar.write(' %s = ' % key.upper())
                [incar.write('%.4f ' % x) for x in val]
                incar.write('\n')
        for key, val in self.bool_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if val:
                    incar.write('.TRUE.\n')
                else:
                    incar.write('.FALSE.\n')
        for key, val in self.special_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if key == 'lreal':
                    if isinstance(val, str):
                        incar.write(val + '\n')
                    elif isinstance(val, bool):
                        if val:
                            incar.write('.TRUE.\n')
                        else:
                            incar.write('.FALSE.\n')
        for key, val in self.dict_params.items():
            if val is not None:
                if key == 'ldau_luj':
                    if self.bool_params['ldau'] is None:
                        self.bool_params['ldau'] = True
                        incar.write(' LDAU = .TRUE.\n')
                    llist = ulist = jlist = ''
                    for symbol in self.symbol_count:
                        luj = val.get(symbol[0], {'L': -1, 'U': 0.0, 'J': 0.0})
                        llist += ' %i' % luj['L']
                        ulist += ' %.3f' % luj['U']
                        jlist += ' %.3f' % luj['J']
                    incar.write(' LDAUL =%s\n' % llist)
                    incar.write(' LDAUU =%s\n' % ulist)
                    incar.write(' LDAUJ =%s\n' % jlist)
        if self.spinpol and (not magmom_written) and atoms.get_initial_magnetic_moments().any():
            if not self.int_params['ispin']:
                incar.write(' ispin = 2\n'.upper())
            magmom = atoms.get_initial_magnetic_moments()[self.sort]
            if magmom.ndim > 1:
                magmom = [item for sublist in magmom for item in sublist]
            list = [[1, magmom[0]]]
            for n in range(1, len(magmom)):
                if magmom[n] == magmom[n - 1]:
                    list[-1][0] += 1
                else:
                    list.append([1, magmom[n]])
            incar.write(' magmom = '.upper())
            [incar.write('%i*%.4f ' % (mom[0], mom[1])) for mom in list]
            incar.write('\n')
        custom_kv_pairs = p.get('custom')
        for key, value in custom_kv_pairs.items():
            incar.write(' {} = {}  # <Custom ASE key>\n'.format(key.upper(), value))
        incar.close()

    def write_kpoints(self, atoms=None, directory='./', **kwargs):
        """Writes the KPOINTS file."""
        if atoms is None:
            atoms = self.atoms
        if self.float_params['kspacing'] is not None:
            if self.float_params['kspacing'] > 0:
                return
            else:
                raise ValueError('KSPACING value {0} is not allowable. Please use None or a positive number.'.format(self.float_params['kspacing']))
        p = self.input_params
        with open(join(directory, 'KPOINTS'), 'w') as kpoints:
            kpoints.write('KPOINTS created by Atomic Simulation Environment\n')
            if isinstance(p['kpts'], dict):
                p['kpts'] = kpts2ndarray(p['kpts'], atoms=atoms)
                p['reciprocal'] = True
            shape = np.array(p['kpts']).shape
            if shape == ():
                p['kpts'] = [p['kpts']]
                shape = (1,)
            if len(shape) == 1:
                kpoints.write('0\n')
                if shape == (1,):
                    kpoints.write('Auto\n')
                elif p['gamma']:
                    kpoints.write('Gamma\n')
                else:
                    kpoints.write('Monkhorst-Pack\n')
                [kpoints.write('%i ' % kpt) for kpt in p['kpts']]
                kpoints.write('\n0 0 0\n')
            elif len(shape) == 2:
                kpoints.write('%i \n' % len(p['kpts']))
                if p['reciprocal']:
                    kpoints.write('Reciprocal\n')
                else:
                    kpoints.write('Cartesian\n')
                for n in range(len(p['kpts'])):
                    [kpoints.write('%f ' % kpt) for kpt in p['kpts'][n]]
                    if shape[1] == 4:
                        kpoints.write('\n')
                    elif shape[1] == 3:
                        kpoints.write('1.0 \n')

    def write_potcar(self, suffix='', directory='./'):
        """Writes the POTCAR file."""
        with open(join(directory, 'POTCAR' + suffix), 'w') as potfile:
            for filename in self.ppp_list:
                with open_potcar(filename=filename) as ppp_file:
                    for line in ppp_file:
                        potfile.write(line)

    def write_sort_file(self, directory='./'):
        """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""
        file = open(join(directory, 'ase-sort.dat'), 'w')
        for n in range(len(self.sort)):
            file.write('%5i %5i \n' % (self.sort[n], self.resort[n]))

    def read_incar(self, filename):
        """Method that imports settings from INCAR file.

        Typically named INCAR."""
        self.spinpol = False
        with open(filename, 'r') as fd:
            lines = fd.readlines()
        for line in lines:
            try:
                line = line.replace('*', ' * ')
                line = line.replace('=', ' = ')
                line = line.replace('#', '# ')
                data = line.split()
                if len(data) == 0:
                    continue
                elif data[0][0] in ['#', '!']:
                    continue
                key = data[0].lower()
                if '<Custom ASE key>' in line:
                    value = line.split('=', 1)[1]
                    value = value.split('#', 1)[0].strip()
                    self.input_params['custom'][key] = value
                elif key in float_keys:
                    self.float_params[key] = float(data[2])
                elif key in exp_keys:
                    self.exp_params[key] = float(data[2])
                elif key in string_keys:
                    self.string_params[key] = str(data[2])
                elif key in int_keys:
                    if key == 'ispin':
                        self.int_params[key] = int(data[2])
                        if int(data[2]) == 2:
                            self.spinpol = True
                    else:
                        self.int_params[key] = int(data[2])
                elif key in bool_keys:
                    if 'true' in data[2].lower():
                        self.bool_params[key] = True
                    elif 'false' in data[2].lower():
                        self.bool_params[key] = False
                elif key in list_bool_keys:
                    self.list_bool_params[key] = [_from_vasp_bool(x) for x in _args_without_comment(data[2:])]
                elif key in list_int_keys:
                    self.list_int_params[key] = [int(x) for x in _args_without_comment(data[2:])]
                elif key in list_float_keys:
                    if key == 'magmom':
                        lst = []
                        i = 2
                        while i < len(data):
                            if data[i] in ['#', '!']:
                                break
                            if data[i] == '*':
                                b = lst.pop()
                                i += 1
                                for j in range(int(b)):
                                    lst.append(float(data[i]))
                            else:
                                lst.append(float(data[i]))
                            i += 1
                        self.list_float_params['magmom'] = lst
                        lst = np.array(lst)
                        if self.atoms is not None:
                            self.atoms.set_initial_magnetic_moments(lst[self.resort])
                    else:
                        data = _args_without_comment(data)
                        self.list_float_params[key] = [float(x) for x in data[2:]]
                elif key in special_keys:
                    if key == 'lreal':
                        if 'true' in data[2].lower():
                            self.special_params[key] = True
                        elif 'false' in data[2].lower():
                            self.special_params[key] = False
                        else:
                            self.special_params[key] = data[2]
            except KeyError:
                raise IOError('Keyword "%s" in INCAR isnot known by calculator.' % key)
            except IndexError:
                raise IOError('Value missing for keyword "%s".' % key)

    def read_kpoints(self, filename):
        """Read kpoints file, typically named KPOINTS."""
        if self.float_params['kspacing'] is not None:
            return
        with open(filename, 'r') as fd:
            lines = fd.readlines()
        ktype = lines[2].split()[0].lower()[0]
        if ktype in ['g', 'm', 'a']:
            if ktype == 'g':
                self.set(gamma=True)
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            elif ktype == 'a':
                kpts = np.array([int(lines[3].split()[i]) for i in range(1)])
            elif ktype == 'm':
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
        else:
            if ktype in ['c', 'k']:
                self.set(reciprocal=False)
            else:
                self.set(reciprocal=True)
            kpts = np.array([list(map(float, line.split())) for line in lines[3:]])
        self.set(kpts=kpts)

    def read_potcar(self, filename):
        """ Read the pseudopotential XC functional from POTCAR file.
        """
        xc_flag = None
        with open(filename, 'r') as fd:
            for line in fd:
                key = line.split()[0].upper()
                if key == 'LEXCH':
                    xc_flag = line.split()[-1].upper()
                    break
        if xc_flag is None:
            raise ValueError('LEXCH flag not found in POTCAR file.')
        xc_dict = {'PE': 'PBE', '91': 'PW91', 'CA': 'LDA'}
        if xc_flag not in xc_dict.keys():
            raise ValueError('Unknown xc-functional flag found in POTCAR, LEXCH=%s' % xc_flag)
        self.input_params['pp'] = xc_dict[xc_flag]

    def todict(self):
        """Returns a dictionary of all parameters
        that can be used to construct a new calculator object"""
        dict_list = ['float_params', 'exp_params', 'string_params', 'int_params', 'bool_params', 'list_bool_params', 'list_int_params', 'list_float_params', 'special_params', 'dict_params', 'input_params']
        dct = {}
        for item in dict_list:
            dct.update(getattr(self, item))
        dct = {key: value for key, value in dct.items() if value is not None}
        return dct