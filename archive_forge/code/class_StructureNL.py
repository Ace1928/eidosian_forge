from __future__ import annotations
import datetime
import json
import re
import sys
from collections import namedtuple
from io import StringIO
from typing import TYPE_CHECKING
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core.structure import Molecule, Structure
class StructureNL:
    """The Structure Notation Language (SNL, pronounced 'snail') is a container for a pymatgen
    Structure/Molecule object with some additional fields for enhanced provenance.

    It is meant to be imported/exported in a JSON file format with the following structure:
        - sites
        - lattice (optional)
        - about
            - created_at
            - authors
            - projects
            - references
            - remarks
            - data
            - history
    """

    def __init__(self, struct_or_mol, authors, projects=None, references='', remarks=None, data=None, history=None, created_at=None):
        """
        Args:
            struct_or_mol: A pymatgen Structure/Molecule object
            authors: *List* of {"name":'', "email":''} dicts,
                *list* of Strings as 'John Doe <johndoe@gmail.com>',
                or a single String with commas separating authors
            projects: List of Strings ['Project A', 'Project B']
            references: A String in BibTeX format
            remarks: List of Strings ['Remark A', 'Remark B']
            data: A free form dict. Namespaced at the root level with an
                underscore, e.g. {"_materialsproject": <custom data>}
            history: List of dicts - [{'name':'', 'url':'', 'description':{}}]
            created_at: A datetime object.
        """
        self.structure = struct_or_mol
        authors = authors.split(',') if isinstance(authors, str) else authors
        self.authors = [Author.parse_author(a) for a in authors]
        projects = projects or []
        self.projects = [projects] if isinstance(projects, str) else projects
        if not isinstance(references, str):
            raise ValueError('Invalid format for SNL reference! Should be empty string or BibTeX string.')
        if references and (not is_valid_bibtex(references)):
            raise ValueError('Invalid format for SNL reference! Should be BibTeX string.')
        if len(references) > MAX_BIBTEX_CHARS:
            raise ValueError(f'The BibTeX string must be fewer than {MAX_BIBTEX_CHARS} chars, you have {len(references)}')
        self.references = references
        remarks = remarks or []
        self.remarks = [remarks] if isinstance(remarks, str) else remarks
        for remark in self.remarks:
            if len(remark) > 140:
                raise ValueError(f'The remark exceeds the maximum size of 140 characters: {len(remark)}')
        self.data = data or {}
        if not sys.getsizeof(self.data) < MAX_DATA_SIZE:
            raise ValueError(f'The data dict exceeds the maximum size limit of {MAX_DATA_SIZE} bytes (you have {sys.getsizeof(data)})')
        for key in self.data:
            if not key.startswith('_'):
                raise ValueError(f'data must contain properly namespaced data with keys starting with an underscore. key={key!r} does not start with an underscore.')
        history = history or []
        if len(history) > MAX_HNODES:
            raise ValueError(f'A maximum of {MAX_HNODES} History nodes are supported, you have {len(history)}!')
        self.history = [HistoryNode.parse_history_node(h) for h in history]
        if not all((sys.getsizeof(h) < MAX_HNODE_SIZE for h in history)):
            raise ValueError(f'One or more history nodes exceeds the maximum size limit of {MAX_HNODE_SIZE} bytes')
        self.created_at = created_at or datetime.datetime.utcnow()

    def as_dict(self):
        """Returns: MSONable dict."""
        dct = self.structure.as_dict()
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['about'] = {'authors': [a.as_dict() for a in self.authors], 'projects': self.projects, 'references': self.references, 'remarks': self.remarks, 'history': [h.as_dict() for h in self.history], 'created_at': json.loads(json.dumps(self.created_at, cls=MontyEncoder))}
        dct['about'].update(json.loads(json.dumps(self.data, cls=MontyEncoder)))
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Class
        """
        about = dct['about']
        created_at = MontyDecoder().process_decoded(about.get('created_at'))
        data = {k: v for k, v in dct['about'].items() if k.startswith('_')}
        data = MontyDecoder().process_decoded(data)
        structure = Structure.from_dict(dct) if 'lattice' in dct else Molecule.from_dict(dct)
        return cls(structure, about['authors'], projects=about.get('projects'), references=about.get('references', ''), remarks=about.get('remarks'), data=data, history=about.get('history'), created_at=created_at)

    @classmethod
    def from_structures(cls, structures: Sequence[Structure], authors: Sequence[dict[str, str]], projects=None, references='', remarks=None, data=None, histories=None, created_at=None) -> list[Self]:
        """A convenience method for getting a list of StructureNL objects by
        specifying structures and metadata separately. Some of the metadata
        is applied to all of the structures for ease of use.

        Args:
            structures: A list of Structure objects
            authors: *List* of {"name":'', "email":''} dicts,
                *list* of Strings as 'John Doe <johndoe@gmail.com>',
                or a single String with commas separating authors
            projects: List of Strings ['Project A', 'Project B']. This
                applies to all structures.
            references: A String in BibTeX format. Again, this applies to all
                structures.
            remarks: List of Strings ['Remark A', 'Remark B']
            data: A list of free form dict. Namespaced at the root level
                with an underscore, e.g. {"_materialsproject":<custom data>}
                . The length of data should be the same as the list of
                structures if not None.
            histories: List of list of dicts - [[{'name':'', 'url':'',
                'description':{}}], ...] The length of histories should be the
                same as the list of structures if not None.
            created_at: A datetime object
        """
        data = [{}] * len(structures) if data is None else data
        histories = [[]] * len(structures) if histories is None else histories
        snl_list = []
        for idx, struct in enumerate(structures):
            snl = cls(struct, authors, projects=projects, references=references, remarks=remarks, data=data[idx], history=histories[idx], created_at=created_at)
            snl_list.append(snl)
        return snl_list

    def __str__(self):
        return '\n'.join([f'{key}\n{getattr(self, key)}' for key in ('structure', 'authors', 'projects', 'references', 'remarks', 'data', 'history', 'created_at')])

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('structure', 'authors', 'projects', 'references', 'remarks', 'data', 'history', 'created_at')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))