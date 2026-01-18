from __future__ import annotations
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from ruamel.yaml import YAML
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecie, DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import IMolecule, IStructure, Molecule, PeriodicNeighbor, SiteCollection, Structure
from pymatgen.core.units import ArrayWithUnit, FloatWithUnit, Unit
def _load_pmg_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    yaml = YAML()
    for file_path in (SETTINGS_FILE, OLD_SETTINGS_FILE):
        try:
            with open(file_path, encoding='utf-8') as yml_file:
                settings = yaml.load(yml_file) or {}
            break
        except FileNotFoundError:
            continue
        except Exception as exc:
            warnings.warn(f'Error loading {file_path}: {exc}.\nYou may need to reconfigure your yaml file.')
    for key, val in os.environ.items():
        if key.startswith('PMG_'):
            settings[key] = val
        elif key in ('VASP_PSP_DIR', 'MAPI_KEY', 'DEFAULT_FUNCTIONAL'):
            settings[f'PMG_{key}'] = val
    return settings