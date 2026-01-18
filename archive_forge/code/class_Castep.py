import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
class Castep(Calculator):
    """
CASTEP Interface Documentation


Introduction
============

CASTEP_ [1]_ W_ is a software package which uses density functional theory to
provide a good atomic-level description of all manner of materials and
molecules. CASTEP can give information about total energies, forces and
stresses on an atomic system, as well as calculating optimum geometries, band
structures, optical spectra, phonon spectra and much more. It can also perform
molecular dynamics simulations.

The CASTEP calculator interface class offers intuitive access to all CASTEP
settings and most results. All CASTEP specific settings are accessible via
attribute access (*i.e*. ``calc.param.keyword = ...`` or
``calc.cell.keyword = ...``)


Getting Started:
================

Set the environment variables appropriately for your system.

>>> export CASTEP_COMMAND=' ... '
>>> export CASTEP_PP_PATH=' ... '

Note: alternatively to CASTEP_PP_PATH one can set PSPOT_DIR
as CASTEP consults this by default, i.e.

>>> export PSPOT_DIR=' ... '


Running the Calculator
======================

The default initialization command for the CASTEP calculator is

.. class:: Castep(directory='CASTEP', label='castep')

To do a minimal run one only needs to set atoms, this will use all
default settings of CASTEP, meaning LDA, singlepoint, etc..

With a generated *castep_keywords.json* in place all options are accessible
by inspection, *i.e.* tab-completion. This works best when using ``ipython``.
All options can be accessed via ``calc.param.<TAB>`` or ``calc.cell.<TAB>``
and documentation is printed with ``calc.param.<keyword> ?`` or
``calc.cell.<keyword> ?``. All options can also be set directly
using ``calc.keyword = ...`` or ``calc.KEYWORD = ...`` or even
``ialc.KeYwOrD`` or directly as named arguments in the call to the constructor
(*e.g.* ``Castep(task='GeometryOptimization')``).
If using this calculator on a machine without CASTEP, one might choose to copy
a *castep_keywords.json* file generated elsewhere in order to access this
feature: the file will be used if located in the working directory,
*$HOME/.ase/* or *ase/ase/calculators/* within the ASE library. The file should
be generated the first time it is needed, but you can generate a new keywords
file in the currect directory with ``python -m ase.calculators.castep``.

All options that go into the ``.param`` file are held in an ``CastepParam``
instance, while all options that go into the ``.cell`` file and don't belong
to the atoms object are held in an ``CastepCell`` instance. Each instance can
be created individually and can be added to calculators by attribute
assignment, *i.e.* ``calc.param = param`` or ``calc.cell = cell``.

All internal variables of the calculator start with an underscore (_).
All cell attributes that clearly belong into the atoms object are blocked.
Setting ``calc.atoms_attribute`` (*e.g.* ``= positions``) is sent directly to
the atoms object.


Arguments:
==========

=========================  ====================================================
Keyword                    Description
=========================  ====================================================
``directory``              The relative path where all input and output files
                           will be placed. If this does not exist, it will be
                           created.  Existing directories will be moved to
                           directory-TIMESTAMP unless self._rename_existing_dir
                           is set to false.

``label``                  The prefix of .param, .cell, .castep, etc. files.

``castep_command``         Command to run castep. Can also be set via the bash
                           environment variable ``CASTEP_COMMAND``. If none is
                           given or found, will default to ``castep``

``check_castep_version``   Boolean whether to check if the installed castep
                           version matches the version from which the available
                           options were deduced. Defaults to ``False``.

``castep_pp_path``         The path where the pseudopotentials are stored. Can
                           also be set via the bash environment variables
                           ``PSPOT_DIR`` (preferred) and ``CASTEP_PP_PATH``.
                           Will default to the current working directory if
                           none is given or found. Note that pseudopotentials
                           may be generated on-the-fly if they are not found.

``find_pspots``            Boolean whether to search for pseudopotentials in
                           ``<castep_pp_path>`` or not. If activated, files in
                           this directory will be checked for typical names. If
                           files are not found, they will be generated on the
                           fly, depending on the ``_build_missing_pspots``
                           value.  A RuntimeError will be raised in case
                           multiple files per element are found. Defaults to
                           ``False``.
``keyword_tolerance``      Integer to indicate the level of tolerance to apply
                           validation of any parameters set in the CastepCell
                           or CastepParam objects against the ones found in
                           castep_keywords. Levels are as following:

                           0 = no tolerance, keywords not found in
                           castep_keywords will raise an exception

                           1 = keywords not found will be accepted but produce
                           a warning (default)

                           2 = keywords not found will be accepted silently

                           3 = no attempt is made to look for
                           castep_keywords.json at all
``castep_keywords``        Can be used to pass a CastepKeywords object that is
                           then used with no attempt to actually load a 
                           castep_keywords.json file. Most useful for debugging
                           and testing purposes.

=========================  ====================================================


Additional Settings
===================

=========================  ====================================================
Internal Setting           Description
=========================  ====================================================
``_castep_command``        (``=castep``): the actual shell command used to
                           call CASTEP.

``_check_checkfile``       (``=True``): this makes write_param() only
                           write a continue or reuse statement if the
                           addressed .check or .castep_bin file exists in the
                           directory.

``_copy_pspots``           (``=False``): if set to True the calculator will
                           actually copy the needed pseudo-potential (\\*.usp)
                           file, usually it will only create symlinks.

``_link_pspots``           (``=True``): if set to True the calculator will
                           actually will create symlinks to the needed pseudo
                           potentials. Set this option (and ``_copy_pspots``)
                           to False if you rather want to access your pseudo
                           potentials using the PSPOT_DIR environment variable
                           that is read by CASTEP.
                           *Note:* This option has no effect if ``copy_pspots``
                           is True..

``_build_missing_pspots``  (``=True``): if set to True, castep will generate
                           missing pseudopotentials on the fly. If not, a
                           RuntimeError will be raised if not all files were
                           found.

``_export_settings``       (``=True``): if this is set to
                           True, all calculator internal settings shown here
                           will be included in the .param in a comment line (#)
                           and can be read again by merge_param. merge_param
                           can be forced to ignore this directive using the
                           optional argument ``ignore_internal_keys=True``.

``_force_write``           (``=True``): this controls wether the \\*cell and
                           \\*param will be overwritten.

``_prepare_input_only``    (``=False``): If set to True, the calculator will
                           create \\*cell und \\*param file but not
                           start the calculation itself.
                           If this is used to prepare jobs locally
                           and run on a remote cluster it is recommended
                           to set ``_copy_pspots = True``.

``_castep_pp_path``        (``='.'``) : the place where the calculator
                           will look for pseudo-potential files.

``_find_pspots``           (``=False``): if set to True, the calculator will
                           try to find the respective pseudopotentials from
                           <_castep_pp_path>. As long as there are no multiple
                           files per element in this directory, the auto-detect
                           feature should be very robust. Raises a RuntimeError
                           if required files are not unique (multiple files per
                           element). Non existing pseudopotentials will be
                           generated, though this could be dangerous.

``_rename_existing_dir``   (``=True``) : when using a new instance
                           of the calculator, this will move directories out of
                           the way that would be overwritten otherwise,
                           appending a date string.

``_set_atoms``             (``=False``) : setting this to True will overwrite
                           any atoms object previously attached to the
                           calculator when reading a \\.castep file.  By de-
                           fault, the read() function will only create a new
                           atoms object if none has been attached and other-
                           wise try to assign forces etc. based on the atom's
                           positions.  ``_set_atoms=True`` could be necessary
                           if one uses CASTEP's internal geometry optimization
                           (``calc.param.task='GeometryOptimization'``)
                           because then the positions get out of sync.
                           *Warning*: this option is generally not recommended
                           unless one knows one really needs it. There should
                           never be any need, if CASTEP is used as a
                           single-point calculator.

``_track_output``          (``=False``) : if set to true, the interface
                           will append a number to the label on all input
                           and output files, where n is the number of calls
                           to this instance. *Warning*: this setting may con-
                           sume a lot more disk space because of the additio-
                           nal \\*check files.

``_try_reuse``             (``=_track_output``) : when setting this, the in-
                           terface will try to fetch the reuse file from the
                           previous run even if _track_output is True. By de-
                           fault it is equal to _track_output, but may be
                           overridden.

                           Since this behavior may not always be desirable for
                           single-point calculations. Regular reuse for *e.g.*
                           a geometry-optimization can be achieved by setting
                           ``calc.param.reuse = True``.

``_pedantic``              (``=False``) if set to true, the calculator will
                           inform about settings probably wasting a lot of CPU
                           time or causing numerical inconsistencies.

=========================  ====================================================

Special features:
=================


``.dryrun_ok()``
  Runs ``castep_command seed -dryrun`` in a temporary directory return True if
  all variables initialized ok. This is a fast way to catch errors in the
  input. Afterwards _kpoints_used is set.

``.merge_param()``
  Takes a filename or filehandler of a .param file or CastepParam instance and
  merges it into the current calculator instance, overwriting current settings

``.keyword.clear()``
  Can be used on any option like ``calc.param.keyword.clear()`` or
  ``calc.cell.keyword.clear()`` to return to the CASTEP default.

``.initialize()``
  Creates all needed input in the ``_directory``. This can then copied to and
  run in a place without ASE or even python.

``.set_pspot('<library>')``
  This automatically sets the pseudo-potential for all present species to
  ``<Species>_<library>.usp``. Make sure that ``_castep_pp_path`` is set
  correctly. Note that there is no check, if the file actually exists. If it
  doesn't castep will crash! You may want to use ``find_pspots()`` instead.

``.find_pspots(pspot=<library>, suffix=<suffix>)``
  This automatically searches for pseudopotentials of type
  ``<Species>_<library>.<suffix>`` or ``<Species>-<library>.<suffix>`` in
  ``castep_pp_path` (make sure this is set correctly). Note that ``<Species>``
  will be searched for case insensitive.  Regular expressions are accepted, and
  arguments ``'*'`` will be regarded as bash-like wildcards. Defaults are any
  ``<library>`` and any ``<suffix>`` from ``['usp', 'UPF', 'recpot']``. If you
  have well-organized folders with pseudopotentials of one kind, this should
  work with the defaults.

``print(calc)``
  Prints a short summary of the calculator settings and atoms.

``ase.io.castep.read_seed('path-to/seed')``
  Given you have a combination of seed.{param,cell,castep} this will return an
  atoms object with the last ionic positions in the .castep file and all other
  settings parsed from the .cell and .param file. If no .castep file is found
  the positions are taken from the .cell file. The output directory will be
  set to the same directory, only the label is preceded by 'copy_of\\_'  to
  avoid overwriting.

``.set_kpts(kpoints)``
  This is equivalent to initialising the calculator with
  ``calc = Castep(kpts=kpoints)``. ``kpoints`` can be specified in many
  convenient forms: simple Monkhorst-Pack grids can be specified e.g.
  ``(2, 2, 3)`` or ``'2 2 3'``; lists of specific weighted k-points can be
  given in reciprocal lattice coordinates e.g.
  ``[[0, 0, 0, 0.25], [0.25, 0.25, 0.25, 0.75]]``; a dictionary syntax is
  available for more complex requirements e.g.
  ``{'size': (2, 2, 2), 'gamma': True}`` will give a Gamma-centered 2x2x2 M-P
  grid, ``{'density': 10, 'gamma': False, 'even': False}`` will give a mesh
  with density of at least 10 Ang (based on the unit cell of currently-attached
  atoms) with an odd number of points in each direction and avoiding the Gamma
  point.

``.set_bandpath(bandpath)``
  This is equivalent to initialialising the calculator with
  ``calc=Castep(bandpath=bandpath)`` and may be set simultaneously with *kpts*.
  It allows an electronic band structure path to be set up using ASE BandPath
  objects. This enables a band structure calculation to be set up conveniently
  using e.g. calc.set_bandpath(atoms.cell.bandpath().interpolate(npoints=200))

``.band_structure(bandfile=None)``
  Read a band structure from _seedname.bands_ file. This returns an ase
  BandStructure object which may be plotted with e.g.
  ``calc.band_structure().plot()``

Notes/Issues:
==============

* Currently *only* the FixAtoms *constraint* is fully supported for reading and
  writing. There is some experimental support for the FixCartesian constraint.

* There is no support for the CASTEP *unit system*. Units of eV and Angstrom
  are used throughout. In particular when converting total energies from
  different calculators, one should check that the same CODATA_ version is
  used for constants and conversion factors, respectively.

.. _CASTEP: http://www.castep.org/

.. _W: https://en.wikipedia.org/wiki/CASTEP

.. _CODATA: https://physics.nist.gov/cuu/Constants/index.html

.. [1] S. J. Clark, M. D. Segall, C. J. Pickard, P. J. Hasnip, M. J. Probert,
       K. Refson, M. C. Payne Zeitschrift für Kristallographie 220(5-6)
       pp.567- 570 (2005) PDF_.

.. _PDF: http://www.tcm.phy.cam.ac.uk/castep/papers/ZKristallogr_2005.pdf


End CASTEP Interface Documentation
    """
    atoms_keys = ['charges', 'ionic_constraints', 'lattice_abs', 'lattice_cart', 'positions_abs', 'positions_abs_final', 'positions_abs_intermediate', 'positions_frac', 'positions_frac_final', 'positions_frac_intermediate']
    atoms_obj_keys = ['dipole', 'energy_free', 'energy_zero', 'fermi', 'forces', 'nbands', 'positions', 'stress', 'pressure']
    internal_keys = ['_castep_command', '_check_checkfile', '_copy_pspots', '_link_pspots', '_find_pspots', '_build_missing_pspots', '_directory', '_export_settings', '_force_write', '_label', '_prepare_input_only', '_castep_pp_path', '_rename_existing_dir', '_set_atoms', '_track_output', '_try_reuse', '_pedantic']

    def __init__(self, directory='CASTEP', label='castep', castep_command=None, check_castep_version=False, castep_pp_path=None, find_pspots=False, keyword_tolerance=1, castep_keywords=None, **kwargs):
        self.__name__ = 'Castep'
        Calculator.__init__(self)
        from ase.io.castep import write_cell
        self._write_cell = write_cell
        if castep_keywords is None:
            castep_keywords = CastepKeywords(make_param_dict(), make_cell_dict(), [], [], 0)
            if keyword_tolerance < 3:
                try:
                    castep_keywords = import_castep_keywords(castep_command)
                except CastepVersionError as e:
                    if keyword_tolerance == 0:
                        raise e
                    else:
                        warnings.warn(str(e))
        self._kw_tol = keyword_tolerance
        keyword_tolerance = max(keyword_tolerance, 2)
        self.param = CastepParam(castep_keywords, keyword_tolerance=keyword_tolerance)
        self.cell = CastepCell(castep_keywords, keyword_tolerance=keyword_tolerance)
        self._calls = 0
        self._castep_version = castep_keywords.castep_version
        self._warnings = []
        self._error = None
        self._interface_warnings = []
        self._old_atoms = None
        self._old_cell = None
        self._old_param = None
        self._opt = {}
        self._castep_command = get_castep_command(castep_command)
        self._castep_pp_path = get_castep_pp_path(castep_pp_path)
        self._check_checkfile = True
        self._copy_pspots = False
        self._link_pspots = True
        self._find_pspots = find_pspots
        self._build_missing_pspots = True
        self._directory = os.path.abspath(directory)
        self._export_settings = True
        self._force_write = True
        self._label = label
        self._prepare_input_only = False
        self._rename_existing_dir = True
        self._set_atoms = False
        self._track_output = False
        self._try_reuse = False
        self._pedantic = False
        self._seed = None
        self.atoms = None
        self._forces = None
        self._energy_total = None
        self._energy_free = None
        self._energy_0K = None
        self._energy_total_corr = None
        self._eigenvalues = None
        self._efermi = None
        self._ibz_kpts = None
        self._ibz_weights = None
        self._band_structure = None
        self._dispcorr_energy_total = None
        self._dispcorr_energy_free = None
        self._dispcorr_energy_0K = None
        self._spins = None
        self._hirsh_volrat = None
        self._mulliken_charges = None
        self._hirshfeld_charges = None
        self._number_of_cell_constraints = None
        self._output_verbosity = None
        self._stress = None
        self._pressure = None
        self._unit_cell = None
        self._kpoints = None
        self._check_file = None
        self._castep_bin_file = None
        self._cut_off_energy = None
        self._total_time = None
        self._peak_memory = None
        if check_castep_version:
            local_castep_version = get_castep_version(self._castep_command)
            if not hasattr(self, '_castep_version'):
                warnings.warn('No castep version found')
                return
            if not local_castep_version == self._castep_version:
                warnings.warn('The options module was generated from version %s while your are currently using CASTEP version %s' % (self._castep_version, get_castep_version(self._castep_command)))
                self._castep_version = local_castep_version
        for keyword, value in kwargs.items():
            if keyword == 'kpts':
                self.set_kpts(value)
            elif keyword == 'bandpath':
                self.set_bandpath(value)
            elif keyword == 'xc':
                self.xc_functional = value
            elif keyword == 'ecut':
                self.cut_off_energy = value
            else:
                self.__setattr__(keyword, value)

    def band_structure(self, bandfile=None):
        from ase.spectrum.band_structure import BandStructure
        if bandfile is None:
            bandfile = os.path.join(self._directory, self._seed) + '.bands'
        if not os.path.exists(bandfile):
            raise ValueError('Cannot find band file "{}".'.format(bandfile))
        kpts, weights, eigenvalues, efermi = read_bands(bandfile)
        special_points = self.atoms.cell.bandpath(npoints=0).special_points
        bandpath = BandPath(self.atoms.cell, kpts=kpts, special_points=special_points)
        return BandStructure(bandpath, eigenvalues, reference=efermi)

    def set_bandpath(self, bandpath):
        """Set a band structure path from ase.dft.kpoints.BandPath object

        This will set the bs_kpoint_list block with a set of specific points
        determined in ASE. bs_kpoint_spacing will not be used; to modify the
        number of points, consider using e.g. bandpath.resample(density=20) to
        obtain a new dense path.

        Args:
            bandpath (:obj:`ase.dft.kpoints.BandPath` or None):
                Set to None to remove list of band structure points. Otherwise,
                sampling will follow BandPath parameters.

        """

        def clear_bs_keywords():
            bs_keywords = product({'bs_kpoint', 'bs_kpoints'}, {'path', 'path_spacing', 'list', 'mp_grid', 'mp_spacing', 'mp_offset'})
            for bs_tag in bs_keywords:
                setattr(self.cell, '_'.join(bs_tag), None)
        if bandpath is None:
            clear_bs_keywords()
        elif isinstance(bandpath, BandPath):
            clear_bs_keywords()
            self.cell.bs_kpoint_list = [' '.join(map(str, row)) for row in bandpath.kpts]
        else:
            raise TypeError('Band structure path must be an ase.dft.kpoint.BandPath object')

    def set_kpts(self, kpts):
        """Set k-point mesh/path using a str, tuple or ASE features

        Args:
            kpts (None, tuple, str, dict):

        This method will set the CASTEP parameters kpoints_mp_grid,
        kpoints_mp_offset and kpoints_mp_spacing as appropriate. Unused
        parameters will be set to None (i.e. not included in input files.)

        If kpts=None, all these parameters are set as unused.

        The simplest useful case is to give a 3-tuple of integers specifying
        a Monkhorst-Pack grid. This may also be formatted as a string separated
        by spaces; this is the format used internally before writing to the
        input files.

        A more powerful set of features is available when using a python
        dictionary with the following allowed keys:

        - 'size' (3-tuple of int) mesh of mesh dimensions
        - 'density' (float) for BZ sampling density in points per recip. Ang
          ( kpoint_mp_spacing = 1 / (2pi * density) ). An explicit MP mesh will
          be set to allow for rounding/centering.
        - 'spacing' (float) for BZ sampling density for maximum space between
          sample points in reciprocal space. This is numerically equivalent to
          the inbuilt ``calc.cell.kpoint_mp_spacing``, but will be converted to
          'density' to allow for rounding/centering.
        - 'even' (bool) to round each direction up to the nearest even number;
          set False for odd numbers, leave as None for no odd/even rounding.
        - 'gamma' (bool) to offset the Monkhorst-Pack grid to include
          (0, 0, 0); set False to offset each direction avoiding 0.
        """

        def clear_mp_keywords():
            mp_keywords = product({'kpoint', 'kpoints'}, {'mp_grid', 'mp_offset', 'mp_spacing', 'list'})
            for kp_tag in mp_keywords:
                setattr(self.cell, '_'.join(kp_tag), None)
        if kpts is None:
            clear_mp_keywords()
            pass
        elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], (tuple, list)):
            if not all(map(lambda row: len(row) == 4, kpts)):
                raise ValueError('In explicit kpt list each row should have 4 elements')
            clear_mp_keywords()
            self.cell.kpoint_list = [' '.join(map(str, row)) for row in kpts]
        elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], str):
            if not all(map(lambda row: len(row.split()) == 4, kpts)):
                raise ValueError('In explicit kpt list each row should have 4 elements')
            clear_mp_keywords()
            self.cell.kpoint_list = kpts
        elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], int):
            if len(kpts) != 3:
                raise ValueError('Monkhorst-pack grid should have 3 values')
            clear_mp_keywords()
            self.cell.kpoint_mp_grid = '%d %d %d' % tuple(kpts)
        elif isinstance(kpts, str):
            self.set_kpts([int(x) for x in kpts.split()])
        elif isinstance(kpts, dict):
            kpts = kpts.copy()
            if kpts.get('spacing') is not None and kpts.get('density') is not None:
                raise ValueError('Cannot set kpts spacing and density simultaneously.')
            else:
                if kpts.get('spacing') is not None:
                    kpts = kpts.copy()
                    spacing = kpts.pop('spacing')
                    kpts['density'] = 1 / (np.pi * spacing)
                clear_mp_keywords()
                size, offsets = kpts2sizeandoffsets(atoms=self.atoms, **kpts)
                self.cell.kpoint_mp_grid = '%d %d %d' % tuple(size)
                self.cell.kpoint_mp_offset = '%f %f %f' % tuple(offsets)
        elif hasattr(kpts, '__iter__'):
            self.set_kpts(list(kpts))
        else:
            raise TypeError('Cannot interpret kpts of this type')

    def todict(self, skip_default=True):
        """Create dict with settings of .param and .cell"""
        dct = {}
        dct['param'] = self.param.get_attr_dict()
        dct['cell'] = self.cell.get_attr_dict()
        return dct

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        return compare_atoms(self._old_atoms, atoms)

    def _castep_find_last_record(self, castep_file):
        """Checks wether a given castep file has a regular
        ending message following the last banner message. If this
        is the case, the line number of the last banner is message
        is return, otherwise False.

        returns (record_start, record_end, end_found, last_record_complete)
        """
        if isinstance(castep_file, str):
            castep_file = paropen(castep_file, 'r')
            file_opened = True
        else:
            file_opened = False
        record_starts = []
        while True:
            line = castep_file.readline()
            if 'Welcome' in line and 'CASTEP' in line:
                record_starts = [castep_file.tell()] + record_starts
            if not line:
                break
        if record_starts == []:
            warnings.warn('Could not find CASTEP label in result file: %s. Are you sure this is a .castep file?' % castep_file)
            return
        end_found = False
        record_end = -1
        for record_nr, record_start in enumerate(record_starts):
            castep_file.seek(record_start)
            while True:
                line = castep_file.readline()
                if not line:
                    break
                if 'warn' in line.lower():
                    self._warnings.append(line)
                if 'Finalisation time   =' in line:
                    end_found = True
                    record_end = castep_file.tell()
                    break
            if end_found:
                break
        if file_opened:
            castep_file.close()
        if end_found:
            if record_nr == 0:
                return (record_start, record_end, True, True)
            else:
                return (record_start, record_end, True, False)
        else:
            return (0, record_end, False, False)

    def read(self, castep_file=None):
        """Read a castep file into the current instance."""
        _close = True
        if castep_file is None:
            if self._castep_file:
                castep_file = self._castep_file
                out = paropen(castep_file, 'r')
            else:
                warnings.warn('No CASTEP file specified')
                return
            if not os.path.exists(castep_file):
                warnings.warn('No CASTEP file found')
        elif isinstance(castep_file, str):
            out = paropen(castep_file, 'r')
        else:
            out = castep_file
            attributes = ['name', 'seek', 'close', 'readline', 'tell']
            for attr in attributes:
                if not hasattr(out, attr):
                    raise TypeError('"castep_file" is neither str nor valid fileobj')
            castep_file = out.name
            _close = False
        if self._seed is None:
            self._seed = os.path.splitext(os.path.basename(castep_file))[0]
        err_file = '%s.0001.err' % self._seed
        if os.path.exists(err_file):
            err_file = paropen(err_file)
            self._error = err_file.read()
            err_file.close()
        record_start, record_end, end_found, _ = self._castep_find_last_record(out)
        if not end_found:
            warnings.warn('No regular end found in %s file. %s' % (castep_file, self._error))
            if _close:
                out.close()
            return
        n_cell_const = 0
        forces = []
        stress = np.zeros([3, 3])
        hirsh_volrat = []
        spin_polarized = False
        calculate_hirshfeld = False
        mulliken_analysis = False
        hirshfeld_analysis = False
        kpoints = None
        positions_frac_list = []
        out.seek(record_start)
        while True:
            try:
                _line_start = out.tell()
                line = out.readline()
                if not line or out.tell() > record_end:
                    break
                elif 'Hirshfeld Analysis' in line:
                    hirshfeld_charges = []
                    hirshfeld_analysis = True
                    line = out.readline()
                    line = out.readline()
                    if 'Charge' in line:
                        line = out.readline()
                        while True:
                            line = out.readline()
                            fields = line.split()
                            if len(fields) == 1:
                                break
                            else:
                                hirshfeld_charges.append(float(fields[-1]))
                elif 'stress calculation' in line:
                    if line.split()[-1].strip() == 'on':
                        self.param.calculate_stress = True
                elif 'basis set accuracy' in line:
                    self.param.basis_precision = line.split()[-1]
                elif 'plane wave basis set cut-off' in line:
                    cutoff = float(line.split()[-2])
                    self._cut_off_energy = cutoff
                    if self.param.basis_precision.value is None:
                        self.param.cut_off_energy = cutoff
                elif 'total energy / atom convergence tol.' in line:
                    elec_energy_tol = float(line.split()[-2])
                    self.param.elec_energy_tol = elec_energy_tol
                elif 'convergence tolerance window' in line:
                    elec_convergence_win = int(line.split()[-2])
                    self.param.elec_convergence_win = elec_convergence_win
                elif re.match('\\sfinite basis set correction\\s*:', line):
                    finite_basis_corr = line.split()[-1]
                    fbc_possibilities = {'none': 0, 'manual': 1, 'automatic': 2}
                    fbc = fbc_possibilities[finite_basis_corr]
                    self.param.finite_basis_corr = fbc
                elif 'Treating system as non-metallic' in line:
                    self.param.fix_occupancy = True
                elif 'max. number of SCF cycles:' in line:
                    max_no_scf = float(line.split()[-1])
                    self.param.max_scf_cycles = max_no_scf
                elif 'density-mixing scheme' in line:
                    mixing_scheme = line.split()[-1]
                    self.param.mixing_scheme = mixing_scheme
                elif 'dump wavefunctions every' in line:
                    no_dump_cycles = float(line.split()[-3])
                    self.param.num_dump_cycles = no_dump_cycles
                elif 'optimization strategy' in line:
                    lspl = line.split(':')
                    if lspl[0].strip() != 'optimization strategy':
                        continue
                    if 'memory' in line:
                        self.param.opt_strategy = 'Memory'
                    if 'speed' in line:
                        self.param.opt_strategy = 'Speed'
                elif 'calculation limited to maximum' in line:
                    calc_limit = float(line.split()[-2])
                    self.param.run_time = calc_limit
                elif 'type of calculation' in line:
                    lspl = line.split(':')
                    if lspl[0].strip() != 'type of calculation':
                        continue
                    calc_type = lspl[-1]
                    calc_type = re.sub('\\s+', ' ', calc_type)
                    calc_type = calc_type.strip()
                    if calc_type != 'single point energy':
                        calc_type_possibilities = {'geometry optimization': 'GeometryOptimization', 'band structure': 'BandStructure', 'molecular dynamics': 'MolecularDynamics', 'optical properties': 'Optics', 'phonon calculation': 'Phonon', 'E-field calculation': 'Efield', 'Phonon followed by E-field': 'Phonon+Efield', 'transition state search': 'TransitionStateSearch', 'Magnetic Resonance': 'MagRes', 'Core level spectra': 'Elnes', 'Electronic Spectroscopy': 'ElectronicSpectroscopy'}
                        ctype = calc_type_possibilities[calc_type]
                        self.param.task = ctype
                elif 'using functional' in line:
                    used_functional = line.split(':')[-1]
                    used_functional = re.sub('\\s+', ' ', used_functional)
                    used_functional = used_functional.strip()
                    if used_functional != 'Local Density Approximation':
                        used_functional_possibilities = {'Perdew Wang (1991)': 'PW91', 'Perdew Burke Ernzerhof': 'PBE', 'revised Perdew Burke Ernzerhof': 'RPBE', 'PBE with Wu-Cohen exchange': 'WC', 'PBE for solids (2008)': 'PBESOL', 'Hartree-Fock': 'HF', 'Hartree-Fock +': 'HF-LDA', 'Screened Hartree-Fock': 'sX', 'Screened Hartree-Fock + ': 'sX-LDA', 'hybrid PBE0': 'PBE0', 'hybrid B3LYP': 'B3LYP', 'hybrid HSE03': 'HSE03', 'hybrid HSE06': 'HSE06'}
                        used_func = used_functional_possibilities[used_functional]
                        self.param.xc_functional = used_func
                elif 'output verbosity' in line:
                    iprint = int(line.split()[-1][1])
                    if int(iprint) != 1:
                        self.param.iprint = iprint
                elif 'treating system as spin-polarized' in line:
                    spin_polarized = True
                    self.param.spin_polarized = spin_polarized
                elif 'treating system as non-spin-polarized' in line:
                    spin_polarized = False
                elif 'Number of kpoints used' in line:
                    kpoints = int(line.split('=')[-1].strip())
                elif 'Unit Cell' in line:
                    lattice_real = []
                    lattice_reci = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 6:
                            break
                    for i in range(3):
                        lattice_real.append([float(f) for f in fields[0:3]])
                        lattice_reci.append([float(f) for f in fields[3:7]])
                        line = out.readline()
                        fields = line.split()
                elif 'Cell Contents' in line:
                    while True:
                        line = out.readline()
                        if 'Total number of ions in cell' in line:
                            n_atoms = int(line.split()[7])
                        if 'Total number of species in cell' in line:
                            int(line.split()[7])
                        fields = line.split()
                        if len(fields) == 0:
                            break
                elif 'Fractional coordinates of atoms' in line:
                    species = []
                    custom_species = None
                    positions_frac = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 7:
                            break
                    for n in range(n_atoms):
                        spec_custom = fields[1].split(':', 1)
                        elem = spec_custom[0]
                        if len(spec_custom) > 1 and custom_species is None:
                            custom_species = list(species)
                        species.append(elem)
                        if custom_species is not None:
                            custom_species.append(fields[1])
                        positions_frac.append([float(s) for s in fields[3:6]])
                        line = out.readline()
                        fields = line.split()
                    positions_frac_list.append(positions_frac)
                elif 'Files used for pseudopotentials' in line:
                    while True:
                        line = out.readline()
                        if 'Pseudopotential generated on-the-fly' in line:
                            continue
                        fields = line.split()
                        if len(fields) >= 2:
                            elem, pp_file = fields
                            self.cell.species_pot = (elem, pp_file)
                        else:
                            break
                elif 'k-Points For BZ Sampling' in line:
                    while True:
                        line = out.readline()
                        if not line.strip():
                            break
                        if 'MP grid size for SCF calculation' in line:
                            break
                elif 'Symmetry and Constraints' in line:
                    out.seek(_line_start)
                    self.read_symops(castep_castep=out)
                elif 'Number of cell constraints' in line:
                    n_cell_const = int(line.split()[4])
                elif 'Final energy' in line:
                    self._energy_total = float(line.split()[-2])
                elif 'Final free energy' in line:
                    self._energy_free = float(line.split()[-2])
                elif 'NB est. 0K energy' in line:
                    self._energy_0K = float(line.split()[-2])
                elif 'Total energy corrected for finite basis set' in line:
                    self._energy_total_corr = float(line.split()[-2])
                elif 'Dispersion corrected final energy' in line:
                    self._dispcorr_energy_total = float(line.split()[-2])
                elif 'Dispersion corrected final free energy' in line:
                    self._dispcorr_energy_free = float(line.split()[-2])
                elif 'dispersion corrected est. 0K energy' in line:
                    self._dispcorr_energy_0K = float(line.split()[-2])
                elif '******************** Forces *********************' in line or '************** Symmetrised Forces ***************' in line or '************** Constrained Symmetrised Forces **************' in line or ('******************** Constrained Forces ********************' in line) or ('******************* Unconstrained Forces *******************' in line):
                    fix = []
                    fix_cart = []
                    forces = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 7:
                            break
                    for n in range(n_atoms):
                        consd = np.array([0, 0, 0])
                        fxyz = [0, 0, 0]
                        for i, force_component in enumerate(fields[-4:-1]):
                            if force_component.count("(cons'd)") > 0:
                                consd[i] = 1
                            fxyz[i] = float(force_component.replace("(cons'd)", ''))
                        if consd.all():
                            fix.append(n)
                        elif consd.any():
                            fix_cart.append(FixCartesian(n, consd))
                        forces.append(fxyz)
                        line = out.readline()
                        fields = line.split()
                elif 'Hirshfeld / free atomic volume :' in line:
                    calculate_hirshfeld = True
                    hirsh_volrat = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 1:
                            break
                    for n in range(n_atoms):
                        hirsh_atom = float(fields[0])
                        hirsh_volrat.append(hirsh_atom)
                        while True:
                            line = out.readline()
                            if 'Hirshfeld / free atomic volume :' in line or 'Hirshfeld Analysis' in line:
                                break
                        line = out.readline()
                        fields = line.split()
                elif '***************** Stress Tensor *****************' in line or '*********** Symmetrised Stress Tensor ***********' in line:
                    stress = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 6:
                            break
                    for n in range(3):
                        stress.append([float(s) for s in fields[2:5]])
                        line = out.readline()
                        fields = line.split()
                    line = out.readline()
                    if 'Pressure:' in line:
                        self._pressure = float(line.split()[-2]) * units.GPa
                elif 'BFGS: starting iteration' in line or 'BFGS: improving iteration' in line:
                    if n_cell_const < 6:
                        lattice_real = []
                        lattice_reci = []
                    if species:
                        prev_species = deepcopy(species)
                    if positions_frac:
                        prev_positions_frac = deepcopy(positions_frac)
                    species = []
                    positions_frac = []
                    forces = []
                    stress = np.zeros([3, 3])
                elif 'Atomic Populations' in line:
                    mulliken_charges = []
                    spins = []
                    mulliken_analysis = True
                    line = out.readline()
                    line = out.readline()
                    if 'Charge' in line:
                        line = out.readline()
                        while True:
                            line = out.readline()
                            fields = line.split()
                            if len(fields) == 1:
                                break
                            if spin_polarized:
                                if len(fields) != 7:
                                    spins.append(float(fields[-1]))
                                    mulliken_charges.append(float(fields[-2]))
                            else:
                                mulliken_charges.append(float(fields[-1]))
                elif 'warn' in line.lower():
                    self._warnings.append(line)
                elif 'Total time' in line:
                    pattern = '.*=\\s*([\\d\\.]+) s'
                    self._total_time = float(re.search(pattern, line).group(1))
                elif 'Peak Memory Use' in line:
                    pattern = '.*=\\s*([\\d]+) kB'
                    self._peak_memory = int(re.search(pattern, line).group(1))
            except Exception as exception:
                sys.stderr.write(line + '|-> line triggered exception: ' + str(exception))
                raise
        if _close:
            out.close()
        if not positions_frac:
            positions_frac = prev_positions_frac
        if not species:
            species = prev_species
        if not spin_polarized:
            spins = np.zeros(len(positions_frac))
        positions_frac_atoms = np.array(positions_frac)
        forces_atoms = np.array(forces)
        spins_atoms = np.array(spins)
        if mulliken_analysis:
            mulliken_charges_atoms = np.array(mulliken_charges)
        else:
            mulliken_charges_atoms = np.zeros(len(positions_frac))
        if hirshfeld_analysis:
            hirshfeld_charges_atoms = np.array(hirshfeld_charges)
        else:
            hirshfeld_charges_atoms = None
        if calculate_hirshfeld:
            hirsh_atoms = np.array(hirsh_volrat)
        else:
            hirsh_atoms = np.zeros_like(spins)
        if self.atoms and (not self._set_atoms):
            atoms_assigned = [False] * len(self.atoms)
            positions_frac_castep = np.array(positions_frac_list[-1])
            forces_castep = np.array(forces)
            hirsh_castep = np.array(hirsh_volrat)
            spins_castep = np.array(spins)
            mulliken_charges_castep = np.array(mulliken_charges_atoms)
            for iase in range(n_atoms):
                for icastep in range(n_atoms):
                    if species[icastep] == self.atoms[iase].symbol and (not atoms_assigned[icastep]):
                        positions_frac_atoms[iase] = positions_frac_castep[icastep]
                        forces_atoms[iase] = np.array(forces_castep[icastep])
                        if iprint > 1 and calculate_hirshfeld:
                            hirsh_atoms[iase] = np.array(hirsh_castep[icastep])
                        if spin_polarized:
                            spins_atoms[iase] = np.array(spins_castep[icastep])
                        mulliken_charges_atoms[iase] = np.array(mulliken_charges_castep[icastep])
                        atoms_assigned[icastep] = True
                        break
            if not all(atoms_assigned):
                not_assigned = [i for i, assigned in zip(range(len(atoms_assigned)), atoms_assigned) if not assigned]
                warnings.warn('%s atoms not assigned.  DEBUGINFO: The following atoms where not assigned: %s' % (atoms_assigned.count(False), not_assigned))
            else:
                self.atoms.set_scaled_positions(positions_frac_atoms)
        else:
            if self.atoms:
                constraints = self.atoms.constraints
            else:
                constraints = []
            atoms = ase.atoms.Atoms(species, cell=lattice_real, constraint=constraints, pbc=True, scaled_positions=positions_frac)
            if custom_species is not None:
                atoms.new_array('castep_custom_species', np.array(custom_species))
            if self.param.spin_polarized:
                atoms.set_initial_magnetic_moments(magmoms=spins_atoms)
            if mulliken_analysis:
                atoms.set_initial_charges(charges=mulliken_charges_atoms)
            atoms.calc = self
        self._kpoints = kpoints
        self._forces = forces_atoms
        self._stress = np.array(stress) * units.GPa
        self._hirsh_volrat = hirsh_atoms
        self._spins = spins_atoms
        self._mulliken_charges = mulliken_charges_atoms
        self._hirshfeld_charges = hirshfeld_charges_atoms
        if self._warnings:
            warnings.warn('WARNING: %s contains warnings' % castep_file)
            for warning in self._warnings:
                warnings.warn(warning)
        self._warnings = []
        bands_file = castep_file[:-7] + '.bands'
        if self.param.task.value is not None and self.param.task.value.lower() == 'bandstructure':
            self._band_structure = self.band_structure(bandfile=bands_file)
        else:
            try:
                self._ibz_kpts, self._ibz_weights, self._eigenvalues, self._efermi = read_bands(filename=bands_file)
            except FileNotFoundError:
                warnings.warn('Could not load .bands file, eigenvalues and Fermi energy are unknown')

    def read_symops(self, castep_castep=None):
        """Read all symmetry operations used from a .castep file."""
        if castep_castep is None:
            castep_castep = self._seed + '.castep'
        if isinstance(castep_castep, str):
            if not os.path.isfile(castep_castep):
                warnings.warn('Warning: CASTEP file %s not found!' % castep_castep)
            f = paropen(castep_castep, 'r')
            _close = True
        else:
            f = castep_castep
            attributes = ['name', 'readline', 'close']
            for attr in attributes:
                if not hasattr(f, attr):
                    raise TypeError('read_castep_castep_symops: castep_castep is not of type str nor valid fileobj!')
            castep_castep = f.name
            _close = False
        while True:
            line = f.readline()
            if not line:
                return
            if 'output verbosity' in line:
                iprint = line.split()[-1][1]
                if int(iprint) != 1:
                    self.param.iprint = iprint
            if 'Symmetry and Constraints' in line:
                break
        if self.param.iprint.value is None or int(self.param.iprint.value) < 2:
            self._interface_warnings.append('Warning: No symmetryoperations could be read from %s (iprint < 2).' % f.name)
            return
        while True:
            line = f.readline()
            if not line:
                break
            if 'Number of symmetry operations' in line:
                nsym = int(line.split()[5])
                symmetry_operations = []
                for _ in range(nsym):
                    rotation = []
                    displacement = []
                    while True:
                        if 'rotation' in f.readline():
                            break
                    for _ in range(3):
                        line = f.readline()
                        rotation.append([float(r) for r in line.split()[1:4]])
                    while True:
                        if 'displacement' in f.readline():
                            break
                    line = f.readline()
                    displacement = [float(d) for d in line.split()[1:4]]
                    symop = {'rotation': rotation, 'displacement': displacement}
                    self.symmetry_ops = symop
                self.symmetry = symmetry_operations
                warnings.warn('Symmetry operations successfully read from %s. %s' % (f.name, self.cell.symmetry_ops))
                break
        if _close:
            f.close()

    def get_hirsh_volrat(self):
        """
        Return the Hirshfeld volumes.
        """
        return self._hirsh_volrat

    def get_spins(self):
        """
        Return the spins from a plane-wave Mulliken analysis.
        """
        return self._spins

    def get_mulliken_charges(self):
        """
        Return the charges from a plane-wave Mulliken analysis.
        """
        return self._mulliken_charges

    def get_hirshfeld_charges(self):
        """
        Return the charges from a Hirshfeld analysis.
        """
        return self._hirshfeld_charges

    def get_total_time(self):
        """
        Return the total runtime
        """
        return self._total_time

    def get_peak_memory(self):
        """
        Return the peak memory usage
        """
        return self._peak_memory

    def set_label(self, label):
        """The label is part of each seed, which in turn is a prefix
        in each CASTEP related file.
        """
        self._label = label

    def set_pspot(self, pspot, elems=None, notelems=None, clear=True, suffix='usp'):
        """Quickly set all pseudo-potentials: Usually CASTEP psp are named
        like <Elem>_<pspot>.<suffix> so this function function only expects
        the <LibraryName>. It then clears any previous pseudopotential
        settings apply the one with <LibraryName> for each element in the
        atoms object. The optional elems and notelems arguments can be used
        to exclusively assign to some species, or to exclude with notelemens.

        Parameters ::

            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp): PP file suffix
        """
        if self._find_pspots:
            if self._pedantic:
                warnings.warn('Warning: <_find_pspots> = True. Do you really want to use `set_pspots()`? This does not check whether the PP files exist. You may rather want to use `find_pspots()` with the same <pspot>.')
        if clear and (not elems) and (not notelems):
            self.cell.species_pot.clear()
        for elem in set(self.atoms.get_chemical_symbols()):
            if elems is not None and elem not in elems:
                continue
            if notelems is not None and elem in notelems:
                continue
            self.cell.species_pot = (elem, '%s_%s.%s' % (elem, pspot, suffix))

    def find_pspots(self, pspot='.+', elems=None, notelems=None, clear=True, suffix='(usp|UPF|recpot)'):
        """Quickly find and set all pseudo-potentials by searching in
        castep_pp_path:

        This one is more flexible than set_pspots, and also checks if the files
        are actually available from the castep_pp_path.

        Essentially, the function parses the filenames in <castep_pp_path> and
        does a regex matching. The respective pattern is:

            r"^(<elem>|<elem.upper()>|elem.lower()>(_|-)<pspot>\\.<suffix>$"

        In most cases, it will be sufficient to not specify anything, if you
        use standard CASTEP USPPs with only one file per element in the
        <castep_pp_path>.

        The function raises a `RuntimeError` if there is some ambiguity
        (multiple files per element).

        Parameters ::

            - pspots ('.+') : as defined above, will be a wildcard if not
                              specified.
            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp|UPF|recpot): PP file suffix
        """
        if clear and (not elems) and (not notelems):
            self.cell.species_pot.clear()
        if not os.path.isdir(self._castep_pp_path):
            if self._pedantic:
                warnings.warn('Cannot search directory: {} Folder does not exist'.format(self._castep_pp_path))
            return
        if pspot == '*':
            pspot = '.*'
        if suffix == '*':
            suffix = '.*'
        if pspot == '*':
            pspot = '.*'
        pattern = '^({elem}|{elem_upper}|{elem_lower})(_|-){pspot}\\.{suffix}$'
        for elem in set(self.atoms.get_chemical_symbols()):
            if elems is not None and elem not in elems:
                continue
            if notelems is not None and elem in notelems:
                continue
            p = pattern.format(elem=elem, elem_upper=elem.upper(), elem_lower=elem.lower(), pspot=pspot, suffix=suffix)
            pps = []
            for f in os.listdir(self._castep_pp_path):
                if re.match(p, f):
                    pps.append(f)
            if not pps:
                if self._pedantic:
                    warnings.warn('Pseudopotential for species {} not found!'.format(elem))
            elif not len(pps) == 1:
                raise RuntimeError('Pseudopotential for species {} not unique!\n'.format(elem) + 'Found the following files in {}\n'.format(self._castep_pp_path) + '\n'.join(['    {}'.format(pp) for pp in pps]) + '\nConsider a stricter search pattern in `find_pspots()`.')
            else:
                self.cell.species_pot = (elem, pps[0])

    @property
    def name(self):
        """Return the name of the calculator (string).  """
        return self.__name__

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name == 'forces':
            return self.get_forces(atoms)
        elif name == 'energy':
            return self.get_potential_energy(atoms)
        elif name == 'stress':
            return self.get_stress(atoms)
        elif name == 'charges':
            return self.get_charges(atoms)
        else:
            raise PropertyNotImplementedError

    @_self_getter
    def get_forces(self, atoms):
        """Run CASTEP calculation if needed and return forces."""
        self.update(atoms)
        return np.array(self._forces)

    @_self_getter
    def get_total_energy(self, atoms):
        """Run CASTEP calculation if needed and return total energy."""
        self.update(atoms)
        return self._energy_total

    @_self_getter
    def get_total_energy_corrected(self, atoms):
        """Run CASTEP calculation if needed and return total energy."""
        self.update(atoms)
        return self._energy_total_corr

    @_self_getter
    def get_free_energy(self, atoms):
        """Run CASTEP calculation if needed and return free energy.
           Only defined with smearing."""
        self.update(atoms)
        return self._energy_free

    @_self_getter
    def get_0K_energy(self, atoms):
        """Run CASTEP calculation if needed and return 0K energy.
           Only defined with smearing."""
        self.update(atoms)
        return self._energy_0K

    @_self_getter
    def get_potential_energy(self, atoms, force_consistent=False):
        """Return the total potential energy."""
        self.update(atoms)
        if force_consistent:
            if self._dispcorr_energy_free is not None:
                return self._dispcorr_energy_free
            else:
                return self._energy_free
        elif self._energy_0K is not None:
            if self._dispcorr_energy_0K is not None:
                return self._dispcorr_energy_0K
            else:
                return self._energy_0K
        elif self._dispcorr_energy_total is not None:
            return self._dispcorr_energy_total
        elif self._energy_total_corr is not None:
            return self._energy_total_corr
        else:
            return self._energy_total

    @_self_getter
    def get_stress(self, atoms):
        """Return the stress."""
        self.update(atoms)
        stress = np.array([self._stress[0, 0], self._stress[1, 1], self._stress[2, 2], self._stress[1, 2], self._stress[0, 2], self._stress[0, 1]])
        return stress

    @_self_getter
    def get_pressure(self, atoms):
        """Return the pressure."""
        self.update(atoms)
        return self._pressure

    @_self_getter
    def get_unit_cell(self, atoms):
        """Return the unit cell."""
        self.update(atoms)
        return self._unit_cell

    @_self_getter
    def get_kpoints(self, atoms):
        """Return the kpoints."""
        self.update(atoms)
        return self._kpoints

    @_self_getter
    def get_number_cell_constraints(self, atoms):
        """Return the number of cell constraints."""
        self.update(atoms)
        return self._number_of_cell_constraints

    @_self_getter
    def get_charges(self, atoms):
        """Run CASTEP calculation if needed and return Mulliken charges."""
        self.update(atoms)
        return np.array(self._mulliken_charges)

    @_self_getter
    def get_magnetic_moments(self, atoms):
        """Run CASTEP calculation if needed and return Mulliken charges."""
        self.update(atoms)
        return np.array(self._spins)

    def set_atoms(self, atoms):
        """Sets the atoms for the calculator and vice versa."""
        atoms.pbc = [True, True, True]
        self.__dict__['atoms'] = atoms.copy()
        self.atoms._calc = self

    def update(self, atoms):
        """Checks if atoms object or calculator changed and
        runs calculation if so.
        """
        if self.calculation_required(atoms):
            self.calculate(atoms)

    def calculation_required(self, atoms, _=None):
        """Checks wether anything changed in the atoms object or CASTEP
        settings since the last calculation using this instance.
        """
        if not atoms == self._old_atoms:
            return True
        if self._old_param is None or self._old_cell is None:
            return True
        if not self.param._options == self._old_param._options:
            return True
        if not self.cell._options == self._old_cell._options:
            return True
        return False

    def calculate(self, atoms):
        """Write all necessary input file and call CASTEP."""
        self.prepare_input_files(atoms, force_write=self._force_write)
        if not self._prepare_input_only:
            self.run()
            self.read()
            self.push_oldstate()

    def push_oldstate(self):
        """This function pushes the current state of the (CASTEP) Atoms object
        onto the previous state. Or in other words after calling this function,
        calculation_required will return False and enquiry functions just
        report the current value, e.g. get_forces(), get_potential_energy().
        """
        self._old_atoms = self.atoms.copy()
        self._old_param = deepcopy(self.param)
        self._old_cell = deepcopy(self.cell)

    def initialize(self, *args, **kwargs):
        """Just an alias for prepar_input_files to comply with standard
        function names in ASE.
        """
        self.prepare_input_files(*args, **kwargs)

    def prepare_input_files(self, atoms=None, force_write=None):
        """Only writes the input .cell and .param files and return
        This can be useful if one quickly needs to prepare input files
        for a cluster where no python or ASE is available. One can than
        upload the file manually and read out the results using
        Castep().read().
        """
        if self.param.reuse.value is None:
            if self._pedantic:
                warnings.warn('You have not set e.g. calc.param.reuse = True. Reusing a previous calculation may save CPU time! The interface will make sure by default, a .check exists. file before adding this statement to the .param file.')
        if self.param.num_dump_cycles.value is None:
            if self._pedantic:
                warnings.warn('You have not set e.g. calc.param.num_dump_cycles = 0. This can save you a lot of disk space. One only needs *wvfn* if electronic convergence is not achieved.')
        from ase.io.castep import write_param
        if atoms is None:
            atoms = self.atoms
        else:
            self.atoms = atoms
        if force_write is None:
            force_write = self._force_write
        if os.path.isdir(self._directory) and self._calls == 0 and self._rename_existing_dir:
            if os.listdir(self._directory) == []:
                os.rmdir(self._directory)
            else:
                ctime = time.localtime(os.lstat(self._directory).st_ctime)
                os.rename(self._directory, '%s.bak-%s' % (self._directory, time.strftime('%Y%m%d-%H%M%S', ctime)))
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory, 509)
        self._fetch_pspots()
        if self._try_reuse and self._calls > 0:
            if os.path.exists(self._abs_path(self._check_file)):
                self.param.reuse = self._check_file
            elif os.path.exists(self._abs_path(self._castep_bin_file)):
                self.param.reuse = self._castep_bin_file
        self._seed = self._build_castep_seed()
        self._check_file = '%s.check' % self._seed
        self._castep_bin_file = '%s.castep_bin' % self._seed
        self._castep_file = self._abs_path('%s.castep' % self._seed)
        self._write_cell(self._abs_path('%s.cell' % self._seed), self.atoms, castep_cell=self.cell, force_write=force_write)
        if self._export_settings:
            interface_options = self._opt
        else:
            interface_options = None
        write_param(self._abs_path('%s.param' % self._seed), self.param, check_checkfile=self._check_checkfile, force_write=force_write, interface_options=interface_options)

    def _build_castep_seed(self):
        """Abstracts to construction of the final castep <seed>
        with and without _tracking_output.
        """
        if self._track_output:
            return '%s-%06d' % (self._label, self._calls)
        else:
            return '%s' % self._label

    def _abs_path(self, path):
        return os.path.join(self._directory, path)

    def run(self):
        """Simply call castep. If the first .err file
        contains text, this will be printed to the screen.
        """
        self._calls += 1
        stdout, stderr = shell_stdouterr('%s %s' % (self._castep_command, self._seed), cwd=self._directory)
        if stdout:
            print('castep call stdout:\n%s' % stdout)
        if stderr:
            print('castep call stderr:\n%s' % stderr)
        err_file = self._abs_path('%s.0001.err' % self._seed)
        if os.path.exists(err_file):
            err_file = open(err_file)
            self._error = err_file.read()
            err_file.close()
        if self._error:
            raise RuntimeError(self._error)

    def __repr__(self):
        """Returns generic, fast to capture representation of
        CASTEP settings along with atoms object.
        """
        expr = ''
        expr += '-----------------Atoms--------------------\n'
        if self.atoms is not None:
            expr += str('%20s\n' % self.atoms)
        else:
            expr += 'None\n'
        expr += '-----------------Param keywords-----------\n'
        expr += str(self.param)
        expr += '-----------------Cell keywords------------\n'
        expr += str(self.cell)
        expr += '-----------------Internal keys------------\n'
        for key in self.internal_keys:
            expr += '%20s : %s\n' % (key, self._opt[key])
        return expr

    def __getattr__(self, attr):
        """___getattr___ gets overloaded to reroute the internal keys
        and to be able to easily store them in in the param so that
        they can be read in again in subsequent calls.
        """
        if attr in self.internal_keys:
            return self._opt[attr]
        if attr in ['__repr__', '__str__']:
            raise AttributeError
        elif attr not in self.__dict__:
            raise AttributeError
        else:
            return self.__dict__[attr]

    def __setattr__(self, attr, value):
        """We overload the settattr method to make value assignment
        as pythonic as possible. Internal values all start with _.
        Value assigment is case insensitive!
        """
        if attr.startswith('_'):
            similars = difflib.get_close_matches(attr, self.internal_keys, cutoff=0.9)
            if attr not in self.internal_keys and similars:
                warnings.warn('Warning: You probably tried one of: %s but typed %s' % (similars, attr))
            if attr in self.internal_keys:
                self._opt[attr] = value
                if attr == '_track_output':
                    if value:
                        self._try_reuse = True
                        if self._pedantic:
                            warnings.warn('You switched _track_output on. This will consume a lot of disk-space. The interface also switched _try_reuse on, which will try to find the last check file. Set _try_reuse = False, if you need really separate calculations')
                    elif '_try_reuse' in self._opt and self._try_reuse:
                        self._try_reuse = False
                        if self._pedantic:
                            warnings.warn('_try_reuse is set to False, too')
            else:
                self.__dict__[attr] = value
            return
        elif attr in ['atoms', 'cell', 'param']:
            if value is not None:
                if attr == 'atoms' and (not isinstance(value, ase.atoms.Atoms)):
                    raise TypeError('%s is not an instance of ase.atoms.Atoms.' % value)
                elif attr == 'cell' and (not isinstance(value, CastepCell)):
                    raise TypeError('%s is not an instance of CastepCell.' % value)
                elif attr == 'param' and (not isinstance(value, CastepParam)):
                    raise TypeError('%s is not an instance of CastepParam.' % value)
            self.__dict__[attr] = value
            return
        elif attr in self.atoms_obj_keys:
            self.atoms.__dict__[attr] = value
            return
        elif attr in self.atoms_keys:
            warnings.warn('Ignoring setings of "%s", since this has to be set through the atoms object' % attr)
            return
        attr = attr.lower()
        if attr not in list(self.cell._options.keys()) + list(self.param._options.keys()):
            if self._kw_tol == 0:
                similars = difflib.get_close_matches(attr, self.cell._options.keys() + self.param._options.keys())
                if similars:
                    raise UserWarning('Option "%s" not known! You mean "%s"?' % (attr, similars[0]))
                else:
                    raise UserWarning('Option "%s" is not known!' % attr)
            else:
                warnings.warn('Option "%s" is not known - please set any new options directly in the .cell or .param objects' % attr)
                return
        if attr in self.param._options.keys():
            comp = 'param'
        elif attr in self.cell._options.keys():
            comp = 'cell'
        else:
            raise UserWarning('Programming error: could not attach the keyword to an input file')
        self.__dict__[comp].__setattr__(attr, value)

    def merge_param(self, param, overwrite=True, ignore_internal_keys=False):
        """Parse a param file and merge it into the current parameters."""
        if isinstance(param, CastepParam):
            for key, option in param._options.items():
                if option.value is not None:
                    self.param.__setattr__(key, option.value)
            return
        elif isinstance(param, str):
            param_file = open(param, 'r')
            _close = True
        else:
            param_file = param
            attributes = ['name', 'closereadlines']
            for attr in attributes:
                if not hasattr(param_file, attr):
                    raise TypeError('"param" is neither CastepParam nor str nor valid fileobj')
            param = param_file.name
            _close = False
        self, int_opts = read_param(fd=param_file, calc=self, get_interface_options=True)
        for k, val in int_opts.items():
            if k in self.internal_keys and (not ignore_internal_keys):
                if val in _tf_table:
                    val = _tf_table[val]
                self._opt[k] = val
        if _close:
            param_file.close()

    def dryrun_ok(self, dryrun_flag='-dryrun'):
        """Starts a CASTEP run with the -dryrun flag [default]
        in a temporary and check wether all variables are initialized
        correctly. This is recommended for every bigger simulation.
        """
        from ase.io.castep import write_param
        temp_dir = tempfile.mkdtemp()
        self._fetch_pspots(temp_dir)
        seed = 'dryrun'
        self._write_cell(os.path.join(temp_dir, '%s.cell' % seed), self.atoms, castep_cell=self.cell)
        if not os.path.isfile(os.path.join(temp_dir, '%s.cell' % seed)):
            warnings.warn('%s.cell not written - aborting dryrun' % seed)
            return
        write_param(os.path.join(temp_dir, '%s.param' % seed), self.param)
        stdout, stderr = shell_stdouterr('%s %s %s' % (self._castep_command, seed, dryrun_flag), cwd=temp_dir)
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        result_file = open(os.path.join(temp_dir, '%s.castep' % seed))
        txt = result_file.read()
        ok_string = '.*DRYRUN finished.*No problems found with input files.*'
        match = re.match(ok_string, txt, re.DOTALL)
        m = re.search('Number of kpoints used =\\s*([0-9]+)', txt)
        if m:
            self._kpoints = int(m.group(1))
        else:
            warnings.warn("Couldn't fetch number of kpoints from dryrun CASTEP file")
        err_file = os.path.join(temp_dir, '%s.0001.err' % seed)
        if match is None and os.path.exists(err_file):
            err_file = open(err_file)
            self._error = err_file.read()
            err_file.close()
        result_file.close()
        shutil.rmtree(temp_dir)
        return match is not None

    def _get_number_in_species(self, at, atoms=None):
        """Return the number of the atoms within the set of it own
        species. If you are an ASE commiter: why not move this into
        ase.atoms.Atoms ?"""
        if atoms is None:
            atoms = self.atoms
        numbers = atoms.get_atomic_numbers()
        n = numbers[at]
        nis = numbers.tolist()[:at + 1].count(n)
        return nis

    def _get_absolute_number(self, species, nic, atoms=None):
        """This is the inverse function to _get_number in species."""
        if atoms is None:
            atoms = self.atoms
        ch = atoms.get_chemical_symbols()
        ch.reverse()
        total_nr = 0
        assert nic > 0, 'Number in species needs to be 1 or larger'
        while True:
            if ch.pop() == species:
                if nic == 1:
                    return total_nr
                nic -= 1
            total_nr += 1

    def _fetch_pspots(self, directory=None):
        """Put all specified pseudo-potentials into the working directory.
        """
        if not os.environ.get('PSPOT_DIR', None) and self._castep_pp_path == os.path.abspath('.'):
            return
        if directory is None:
            directory = self._directory
        if not os.path.isdir(self._castep_pp_path):
            warnings.warn('PSPs directory %s not found' % self._castep_pp_path)
        pspots = {}
        if self._find_pspots:
            self.find_pspots()
        if self.cell.species_pot.value is not None:
            for line in self.cell.species_pot.value.split('\n'):
                line = line.split()
                if line:
                    pspots[line[0]] = line[1]
        for species in self.atoms.get_chemical_symbols():
            if not pspots or species not in pspots.keys():
                if self._build_missing_pspots:
                    if self._pedantic:
                        warnings.warn('Warning: you have no PP specified for %s. CASTEP will now generate an on-the-fly potentials. For sake of numerical consistency and efficiency this is discouraged.' % species)
                else:
                    raise RuntimeError('Warning: you have no PP specified for %s.' % species)
        if self.cell.species_pot.value:
            for species, pspot in pspots.items():
                orig_pspot_file = os.path.join(self._castep_pp_path, pspot)
                cp_pspot_file = os.path.join(directory, pspot)
                if os.path.exists(orig_pspot_file) and (not os.path.exists(cp_pspot_file)):
                    if self._copy_pspots:
                        shutil.copy(orig_pspot_file, directory)
                    elif self._link_pspots:
                        os.symlink(orig_pspot_file, cp_pspot_file)
                    elif self._pedantic:
                        warnings.warn('Warning: PP files have neither been linked nor copied to the working directory. Make sure to set the evironment variable PSPOT_DIR accordingly!')