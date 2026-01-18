from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
class AirssProvider(ResProvider):
    """
    Provides access to the res file as does ResProvider. This class additionally provides
    access to fields in the TITL entry and various other fields found in the REM entries
    that AIRSS puts in the file. Values in the TITL entry that AIRSS could not get end up as 0.
    If the TITL entry is malformed, empty, or missing then attempting to construct this class
    from a res file will raise a ResError.

    While AIRSS supports a number of geometry and energy solvers, CASTEP is the default. As such,
    fetching the information from the REM entries is only supported if AIRSS was used with CASTEP.
    The other properties that get put in the TITL should still be accessible even if CASTEP was
    not used.

    The :attr:`parse_rems` attribute controls whether functions that fail to retrieve information
    from the REM entries should return ``None``. If this is set to ``"strict"``,
    then a ParseError may be raised, but the return value will not be ``None``.
    If it is set to ``"gentle"``, then ``None`` will be returned instead of raising an
    exception. This setting applies to all methods of this class that are typed to return
    an Optional type. Default is ``"gentle"``.
    """
    _date_fmt = re.compile('[MTWFS][a-z]{2}, (\\d{2}) ([A-Z][a-z]{2}) (\\d{4}) (\\d{2}):(\\d{2}):(\\d{2}) ([+-]?\\d{4})')

    def __init__(self, res: Res, parse_rems: Literal['gentle', 'strict']='gentle'):
        """The :func:`from_str` and :func:`from_file` methods should be used instead of constructing this directly."""
        super().__init__(res)
        if self._res.TITL is None:
            raise ResError(f'{type(self).__name__} can only be constructed from a res file with a valid TITL entry.')
        if parse_rems not in ['gentle', 'strict']:
            raise ValueError(f"{parse_rems} not valid, must be either 'gentle' or 'strict'.")
        self._TITL = self._res.TITL
        self.parse_rems = parse_rems

    @classmethod
    def from_str(cls, string: str, parse_rems: Literal['gentle', 'strict']='gentle') -> Self:
        """Construct a Provider from a string."""
        return cls(ResParser._parse_str(string), parse_rems)

    @classmethod
    def from_file(cls, filename: str | Path, parse_rems: Literal['gentle', 'strict']='gentle') -> Self:
        """Construct a Provider from a file."""
        return cls(ResParser._parse_file(filename), parse_rems)

    @classmethod
    def _parse_date(cls, string: str) -> date:
        """Parses a date from a string where the date is in the format typically used by CASTEP."""
        match = cls._date_fmt.search(string)
        if match is None:
            raise ResParseError(f'Could not parse the date from string={string!r}.')
        day, month, year, *_ = match.groups()
        month_num = datetime.datetime.strptime(month, '%b').month
        return datetime.date(int(year), month_num, int(day))

    def _raise_or_none(self, err: ResParseError) -> None:
        if self.parse_rems != 'strict':
            return
        raise err

    def get_run_start_info(self) -> tuple[date, str] | None:
        """
        Retrieves the run start date and the path it was started in from the REM entries.

        Returns:
            tuple[date, str]: (date, path)
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('Run started:'):
                date = self._parse_date(rem)
                path = rem.split()[-1]
                return (date, path)
        self._raise_or_none(ResParseError('Could not find run started information.'))
        return None

    def get_castep_version(self) -> str | None:
        """
        Retrieves the version of CASTEP that the res file was computed with from the REM entries.

        Returns:
            version string
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('CASTEP'):
                srem = rem.split()
                return srem[1][:-1]
        self._raise_or_none(ResParseError('No CASTEP version found in REM'))
        return None

    def get_func_rel_disp(self) -> tuple[str, str, str] | None:
        """
        Retrieves the functional, relativity scheme, and dispersion correction from the REM entries.

        Returns:
            tuple[str, str, str]: (functional, relativity, dispersion)
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('Functional'):
                srem = rem.split()
                return (' '.join(srem[1:4]), srem[5], srem[7])
        self._raise_or_none(ResParseError('Could not find functional, relativity, and dispersion.'))
        return None

    def get_cut_grid_gmax_fsbc(self) -> tuple[float, float, float, str] | None:
        """
        Retrieves the cut-off energy, grid scale, Gmax, and finite basis set correction setting
        from the REM entries.

        Returns:
            tuple[float, float, float, str]: (cut-off, grid scale, Gmax, fsbc)
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('Cut-off'):
                srem = rem.split()
                return (float(srem[1]), float(srem[5]), float(srem[7]), srem[10])
        self._raise_or_none(ResParseError('Could not find line with cut-off energy.'))
        return None

    def get_mpgrid_offset_nkpts_spacing(self) -> tuple[tuple[int, int, int], Vector3D, int, float] | None:
        """
        Retrieves the MP grid, the grid offsets, number of kpoints, and maximum kpoint spacing.

        Returns:
            tuple[tuple[int, int, int], Vector3D, int, float]: (MP grid), (offsets), No. kpts, max spacing)
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('MP grid'):
                srem = rem.split()
                p, q, r = map(int, srem[2:5])
                po, qo, ro = map(float, srem[6:9])
                return ((p, q, r), (po, qo, ro), int(srem[11]), float(srem[13]))
        self._raise_or_none(ResParseError('Could not find line with MP grid.'))
        return None

    def get_airss_version(self) -> tuple[str, date] | None:
        """
        Retrieves the version of AIRSS that was used along with the build date (not compile date).

        Returns:
            tuple[str, date] (version string, date)
        """
        for rem in self._res.REMS:
            if rem.strip().startswith('AIRSS Version'):
                date = self._parse_date(rem)
                v = rem.split()[2]
                return (v, date)
        self._raise_or_none(ResParseError('Could not find line with AIRSS version.'))
        return None

    def _get_compiler(self):
        raise NotImplementedError

    def _get_compile_options(self):
        raise NotImplementedError

    def _get_rng_seeds(self):
        raise NotImplementedError

    def get_pspots(self) -> dict[str, str]:
        """
        Retrieves the OTFG pseudopotential string that can be used to generate the
        pseudopotentials used in the calculation.

        Returns:
            dict[specie, potential]
        """
        pseudo_pots: dict[str, str] = {}
        for rem in self._res.REMS:
            srem = rem.split()
            if len(srem) == 2 and Element.is_valid_symbol(srem[0]):
                k, v = srem
                pseudo_pots[k] = v
        return pseudo_pots

    @property
    def seed(self) -> str:
        """The seed name, typically also the name of the res file."""
        return self._TITL.seed

    @property
    def pressure(self) -> float:
        """Pressure for the run. This is in GPa if CASTEP was used."""
        return self._TITL.pressure

    @property
    def volume(self) -> float:
        """Volume of the structure. This is in cubic Angstroms if CASTEP was used."""
        return self._TITL.volume

    @property
    def energy(self) -> float:
        """Energy of the structure. With CASTEP, this is usually the enthalpy and is in eV."""
        return self._TITL.energy

    @property
    def integrated_spin_density(self) -> float:
        """Corresponds to the last ``Integrated Spin Density`` in the CASTEP file."""
        return self._TITL.integrated_spin_density

    @property
    def integrated_absolute_spin_density(self) -> float:
        """Corresponds to the last ``Integrated |Spin Density|`` in the CASTEP file."""
        return self._TITL.integrated_absolute_spin_density

    @property
    def spacegroup_label(self) -> str:
        """
        The Hermann-Mauguin notation of the spacegroup with ascii characters.
        So no. 225 would be Fm-3m, and no. 194 would be P6_3/mmc.
        """
        return self._TITL.spacegroup_label

    @property
    def appearances(self) -> int:
        """
        This is sometimes the number of times a structure was found in an AIRSS search.
        Using the cryan tool that comes with AIRSS may be a better approach than relying
        on this property.
        """
        return self._TITL.appearances

    @property
    def entry(self) -> ComputedStructureEntry:
        """Get this res file as a ComputedStructureEntry."""
        return ComputedStructureEntry(self.structure, self.energy, data={'rems': self.rems})

    def as_dict(self, verbose: bool=True) -> dict[str, Any]:
        """Get dict with title fields, structure and rems of this AirssProvider."""
        if verbose:
            return super().as_dict()
        return dict(**vars(self._res.TITL), structure=self.structure.as_dict(), rems=self.rems)