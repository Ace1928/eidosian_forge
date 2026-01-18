from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
@staticmethod
def _get_snls_from_resource(json, url, identifier) -> dict[str, StructureNL]:
    snls = {}
    exceptions = set()

    def _sanitize_symbol(symbol):
        if symbol == 'vacancy':
            symbol = DummySpecies('X_vacancy', oxidation_state=None)
        elif symbol == 'X':
            symbol = DummySpecies('X', oxidation_state=None)
        return symbol

    def _get_comp(sp_dict):
        return {_sanitize_symbol(symbol): conc for symbol, conc in zip(sp_dict['chemical_symbols'], sp_dict['concentration'])}
    for data in json['data']:
        try:
            structure = Structure(lattice=data['attributes']['lattice_vectors'], species=[_get_comp(d) for d in data['attributes']['species']], coords=data['attributes']['cartesian_site_positions'], coords_are_cartesian=True)
            namespaced_data = {k: v for k, v in data['attributes'].items() if k.startswith('_') or k not in {'lattice_vectors', 'species', 'cartesian_site_positions'}}
            snl = StructureNL(structure, authors={}, history=[{'name': identifier, 'url': url, 'description': {'id': data['id']}}], data={'_optimade': namespaced_data})
            snls[data['id']] = snl
        except Exception:
            try:
                structure = Structure(lattice=data['attributes']['lattice_vectors'], species=data['attributes']['species_at_sites'], coords=data['attributes']['cartesian_site_positions'], coords_are_cartesian=True)
                namespaced_data = {k: v for k, v in data['attributes'].items() if k.startswith('_') or k not in {'lattice_vectors', 'species', 'cartesian_site_positions'}}
                snl = StructureNL(structure, authors={}, history=[{'name': identifier, 'url': url, 'description': {'id': data['id']}}], data={'_optimade': namespaced_data})
                snls[data['id']] = snl
            except Exception as exc:
                if str(exc) not in exceptions:
                    exceptions.add(str(exc))
    if exceptions:
        _logger.error(f'Failed to parse returned data for {url}: {', '.join(exceptions)}')
    return snls