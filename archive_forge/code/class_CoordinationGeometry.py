from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class CoordinationGeometry:
    """Class used to store the ideal representation of a chemical environment or "coordination geometry"."""
    CSM_SKIP_SEPARATION_PLANE_ALGO = 10.0

    class NeighborsSetsHints:
        """
        Class used to describe neighbors sets hints.

        This allows to possibly get a lower coordination from a capped-like model polyhedron.
        """
        ALLOWED_HINTS_TYPES = ('single_cap', 'double_cap', 'triple_cap')

        def __init__(self, hints_type, options):
            """Constructor for this NeighborsSetsHints.

            Args:
                hints_type: type of hint (single, double or triple cap)
                options: options for the "hinting", e.g. the maximum csm value beyond which no additional
                    neighbors set could be found from a "cap hint".
            """
            if hints_type not in self.ALLOWED_HINTS_TYPES:
                raise ValueError(f'Type {type!r} for NeighborsSetsHints is not allowed')
            self.hints_type = hints_type
            self.options = options

        def hints(self, hints_info):
            """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
            if hints_info['csm'] > self.options['csm_max']:
                return []
            return getattr(self, f'{self.hints_type}_hints')(hints_info)

        def single_cap_hints(self, hints_info):
            """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Single cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
            cap_index_perfect = self.options['cap_index']
            nb_set = hints_info['nb_set']
            permutation = hints_info['permutation']
            nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
            cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[cap_index_perfect]
            new_site_voronoi_indices = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices.remove(cap_voronoi_index)
            return [new_site_voronoi_indices]

        def double_cap_hints(self, hints_info):
            """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Double cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
            first_cap_index_perfect = self.options['first_cap_index']
            second_cap_index_perfect = self.options['second_cap_index']
            nb_set = hints_info['nb_set']
            permutation = hints_info['permutation']
            nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
            first_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[first_cap_index_perfect]
            second_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[second_cap_index_perfect]
            new_site_voronoi_indices1 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices2 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices3 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices1.remove(first_cap_voronoi_index)
            new_site_voronoi_indices2.remove(second_cap_voronoi_index)
            new_site_voronoi_indices3.remove(first_cap_voronoi_index)
            new_site_voronoi_indices3.remove(second_cap_voronoi_index)
            return (new_site_voronoi_indices1, new_site_voronoi_indices2, new_site_voronoi_indices3)

        def triple_cap_hints(self, hints_info):
            """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Triple cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
            first_cap_index_perfect = self.options['first_cap_index']
            second_cap_index_perfect = self.options['second_cap_index']
            third_cap_index_perfect = self.options['third_cap_index']
            nb_set = hints_info['nb_set']
            permutation = hints_info['permutation']
            nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
            first_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[first_cap_index_perfect]
            second_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[second_cap_index_perfect]
            third_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[third_cap_index_perfect]
            new_site_voronoi_indices1 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices2 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices3 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices4 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices5 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices6 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices7 = list(nb_set.site_voronoi_indices)
            new_site_voronoi_indices1.remove(first_cap_voronoi_index)
            new_site_voronoi_indices2.remove(second_cap_voronoi_index)
            new_site_voronoi_indices3.remove(third_cap_voronoi_index)
            new_site_voronoi_indices4.remove(second_cap_voronoi_index)
            new_site_voronoi_indices4.remove(third_cap_voronoi_index)
            new_site_voronoi_indices5.remove(first_cap_voronoi_index)
            new_site_voronoi_indices5.remove(third_cap_voronoi_index)
            new_site_voronoi_indices6.remove(first_cap_voronoi_index)
            new_site_voronoi_indices6.remove(second_cap_voronoi_index)
            new_site_voronoi_indices7.remove(first_cap_voronoi_index)
            new_site_voronoi_indices7.remove(second_cap_voronoi_index)
            new_site_voronoi_indices7.remove(third_cap_voronoi_index)
            return [new_site_voronoi_indices1, new_site_voronoi_indices2, new_site_voronoi_indices3, new_site_voronoi_indices4, new_site_voronoi_indices5, new_site_voronoi_indices6, new_site_voronoi_indices7]

        def as_dict(self):
            """A JSON-serializable dict representation of this NeighborsSetsHints."""
            return {'hints_type': self.hints_type, 'options': self.options}

        @classmethod
        def from_dict(cls, dct: dict) -> Self:
            """Reconstructs the NeighborsSetsHints from its JSON-serializable dict representation."""
            return cls(hints_type=dct['hints_type'], options=dct['options'])

    def __init__(self, mp_symbol, name, alternative_names=None, IUPAC_symbol=None, IUCr_symbol=None, coordination=None, central_site=None, points=None, solid_angles=None, permutations_safe_override=False, deactivate=False, faces=None, edges=None, algorithms=None, equivalent_indices=None, neighbors_sets_hints=None):
        """
        Initializes one "coordination geometry" according to [Pure Appl. Chem., Vol. 79, No. 10, pp. 1779--1799, 2007]
        and [Acta Cryst. A, Vol. 46, No. 1, pp. 1--11, 1990].

        Args:
            mp_symbol: Symbol used internally for the coordination geometry.
            name: Name of the coordination geometry.
            alternative_names: Alternative names for this coordination geometry.
            IUPAC_symbol: The IUPAC symbol of this coordination geometry.
            IUCr_symbol: The IUCr symbol of this coordination geometry.
            coordination: The coordination number of this coordination geometry (number of neighboring atoms).
            central_site: The coordinates of the central site of this coordination geometry.
            points: The list of the coordinates of all the points of this coordination geometry.
            solid_angles: The list of solid angles for each neighbor in this coordination geometry.
            permutations_safe_override: Computes all the permutations if set to True (overrides the plane separation
                algorithms or any other algorithm, for testing purposes)
            deactivate: Whether to deactivate this coordination geometry
            faces: List of the faces with their vertices given in a clockwise or anticlockwise order, for drawing
                purposes.
            edges: List of edges, for drawing purposes.
            algorithms: Algorithms used to identify this coordination geometry.
            equivalent_indices: The equivalent sets of indices in this coordination geometry (can be used to skip
                equivalent permutations that have already been performed).
            neighbors_sets_hints: Neighbors sets hints for this coordination geometry.
        """
        self._mp_symbol = mp_symbol
        self.name = name
        self.alternative_names = alternative_names if alternative_names is not None else []
        self.IUPACsymbol = IUPAC_symbol
        self.IUCrsymbol = IUCr_symbol
        self.coordination = coordination
        self.central_site = np.array(central_site or np.zeros(3))
        self.points = points
        self._solid_angles = solid_angles
        self.permutations_safe_override = permutations_safe_override
        self.deactivate = deactivate
        self._faces = faces
        self._edges = edges
        self._algorithms = algorithms
        if points is not None:
            self.centroid = np.mean(np.array(points), axis=0)
        else:
            self.centroid = None
        self.equivalent_indices = equivalent_indices
        self.neighbors_sets_hints = neighbors_sets_hints
        self._pauling_stability_ratio = None

    def as_dict(self):
        """A JSON-serializable dict representation of this CoordinationGeometry."""
        return {'mp_symbol': self._mp_symbol, 'name': self.name, 'alternative_names': self.alternative_names, 'IUPAC_symbol': self.IUPACsymbol, 'IUCr_symbol': self.IUCrsymbol, 'coordination': self.coordination, 'central_site': [float(xx) for xx in self.central_site], 'points': [[float(xx) for xx in pp] for pp in self.points or []] or None, 'solid_angles': [float(ang) for ang in self._solid_angles or []] or None, 'deactivate': self.deactivate, '_faces': self._faces, '_edges': self._edges, '_algorithms': [algo.as_dict for algo in self._algorithms or []] or None, 'equivalent_indices': self.equivalent_indices, 'neighbors_sets_hints': [nbsh.as_dict() for nbsh in self.neighbors_sets_hints] if self.neighbors_sets_hints is not None else None}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the CoordinationGeometry from its JSON-serializable dict representation.

        Args:
            dct: a JSON-serializable dict representation of a CoordinationGeometry.

        Returns:
            CoordinationGeometry
        """
        return cls(mp_symbol=dct['mp_symbol'], name=dct['name'], alternative_names=dct['alternative_names'], IUPAC_symbol=dct['IUPAC_symbol'], IUCr_symbol=dct['IUCr_symbol'], coordination=dct['coordination'], central_site=dct['central_site'], points=dct['points'], solid_angles=dct['solid_angles'] if 'solid_angles' in dct else [4.0 * np.pi / dct['coordination']] * dct['coordination'], deactivate=dct['deactivate'], faces=dct['_faces'], edges=dct['_edges'], algorithms=[MontyDecoder().process_decoded(algo_d) for algo_d in dct['_algorithms']] if dct['_algorithms'] is not None else None, equivalent_indices=dct.get('equivalent_indices'), neighbors_sets_hints=[cls.NeighborsSetsHints.from_dict(nb_sets_hints) for nb_sets_hints in dct.get('neighbors_sets_hints') or []] or None)

    def __str__(self):
        symbol = ''
        if self.IUPAC_symbol is not None:
            symbol += f' (IUPAC: {self.IUPAC_symbol}'
            if self.IUCr_symbol is not None:
                symbol += f' || IUCr: {self.IUCr_symbol})'
            else:
                symbol += ')'
        elif self.IUCr_symbol is not None:
            symbol += f' (IUCr: {self.IUCr_symbol})'
        outs = [f'Coordination geometry type : {self.name}{symbol}\n', f'  - coordination number : {self.coordination}']
        if self.points is None:
            outs.append('... not yet implemented')
        else:
            outs.append('  - list of points :')
            for pp in self.points:
                outs.append(f'    - {pp}')
        outs.extend(('------------------------------------------------------------', ''))
        return '\n'.join(outs)

    def __repr__(self):
        symbol = ''
        if self.IUPAC_symbol is not None:
            symbol += f' (IUPAC: {self.IUPAC_symbol}'
            if self.IUCr_symbol is not None:
                symbol += f' || IUCr: {self.IUCr_symbol})'
            else:
                symbol += ')'
        elif self.IUCr_symbol is not None:
            symbol += f' (IUCr: {self.IUCr_symbol})'
        outs = [f'Coordination geometry type : {self.name}{symbol}\n', f'  - coordination number : {self.coordination}']
        outs.extend(('------------------------------------------------------------', ''))
        return '\n'.join(outs)

    def __len__(self):
        return self.coordination

    @property
    def distfactor_max(self):
        """The maximum distfactor for the perfect CoordinationGeometry (usually 1.0 for symmetric polyhedrons)."""
        dists = [np.linalg.norm(pp - self.central_site) for pp in self.points]
        return np.max(dists) / np.min(dists)

    @property
    def coordination_number(self):
        """Returns the coordination number of this coordination geometry."""
        return self.coordination

    @property
    def pauling_stability_ratio(self):
        """Returns the theoretical Pauling stability ratio (rC/rA) for this environment."""
        if self._pauling_stability_ratio is None:
            if self.ce_symbol in ['S:1', 'L:2']:
                self._pauling_stability_ratio = 0.0
            else:
                min_dist_anions = 1000000
                min_dist_cation_anion = 1000000
                for ipt1 in range(len(self.points)):
                    pt1 = np.array(self.points[ipt1])
                    min_dist_cation_anion = min(min_dist_cation_anion, np.linalg.norm(pt1 - self.central_site))
                    for ipt2 in range(ipt1 + 1, len(self.points)):
                        pt2 = np.array(self.points[ipt2])
                        min_dist_anions = min(min_dist_anions, np.linalg.norm(pt1 - pt2))
                anion_radius = min_dist_anions / 2
                cation_radius = min_dist_cation_anion - anion_radius
                self._pauling_stability_ratio = cation_radius / anion_radius
        return self._pauling_stability_ratio

    @property
    def mp_symbol(self):
        """Returns the MP symbol of this coordination geometry."""
        return self._mp_symbol

    @property
    def ce_symbol(self):
        """Returns the symbol of this coordination geometry."""
        return self._mp_symbol

    def get_coordination_number(self):
        """Returns the coordination number of this coordination geometry."""
        return self.coordination

    def is_implemented(self) -> bool:
        """Returns True if this coordination geometry is implemented."""
        return bool(self.points)

    def get_name(self):
        """Returns the name of this coordination geometry."""
        return self.name

    @property
    def IUPAC_symbol(self):
        """Returns the IUPAC symbol of this coordination geometry."""
        return self.IUPACsymbol

    @property
    def IUPAC_symbol_str(self):
        """Returns a string representation of the IUPAC symbol of this coordination geometry."""
        return str(self.IUPACsymbol)

    @property
    def IUCr_symbol(self):
        """Returns the IUCr symbol of this coordination geometry."""
        return self.IUCrsymbol

    @property
    def IUCr_symbol_str(self):
        """Returns a string representation of the IUCr symbol of this coordination geometry."""
        return str(self.IUCrsymbol)

    @property
    def number_of_permutations(self):
        """Returns the number of permutations of this coordination geometry."""
        if self.permutations_safe_override:
            return factorial(self.coordination)
        if self.permutations is None:
            return factorial(self.coordination)
        return len(self.permutations)

    def ref_permutation(self, permutation):
        """
        Returns the reference permutation for a set of equivalent permutations.

        Can be useful to skip permutations that have already been performed.

        Args:
            permutation: Current permutation

        Returns:
            Permutation: Reference permutation of the perfect CoordinationGeometry.
        """
        perms = []
        for eqv_indices in self.equivalent_indices:
            perms.append(tuple((permutation[ii] for ii in eqv_indices)))
        perms.sort()
        return perms[0]

    @property
    def algorithms(self):
        """Returns the list of algorithms that are used to identify this coordination geometry."""
        return self._algorithms

    def get_central_site(self):
        """Returns the central site of this coordination geometry."""
        return self.central_site

    def faces(self, sites, permutation=None):
        """
        Returns the list of faces of this coordination geometry. Each face is given as a
        list of its vertices coordinates.
        """
        coords = [site.coords for site in sites] if permutation is None else [sites[ii].coords for ii in permutation]
        return [[coords[ii] for ii in face] for face in self._faces]

    def edges(self, sites, permutation=None, input='sites'):
        """
        Returns the list of edges of this coordination geometry. Each edge is given as a
        list of its end vertices coordinates.
        """
        if input == 'sites':
            coords = [site.coords for site in sites]
        elif input == 'coords':
            coords = sites
        if permutation is not None:
            coords = [coords[ii] for ii in permutation]
        return [[coords[ii] for ii in edge] for edge in self._edges]

    def solid_angles(self, permutation=None):
        """
        Returns the list of "perfect" solid angles Each edge is given as a
        list of its end vertices coordinates.
        """
        if permutation is None:
            return self._solid_angles
        return [self._solid_angles[ii] for ii in permutation]

    def get_pmeshes(self, sites, permutation=None):
        """Returns the pmesh strings used for jmol to show this geometry."""
        pmeshes = []
        _vertices = [site.coords for site in sites] if permutation is None else [sites[ii].coords for ii in permutation]
        _face_centers = []
        n_faces = 0
        for face in self._faces:
            if len(face) in [3, 4]:
                n_faces += 1
            else:
                n_faces += len(face)
            _face_centers.append(np.array([np.mean([_vertices[face_vertex][ii] for face_vertex in face]) for ii in range(3)]))
        out = f'{len(_vertices) + len(_face_centers)}\n'
        for vv in _vertices:
            out += f'{vv[0]:15.8f} {vv[1]:15.8f} {vv[2]:15.8f}\n'
        for fc in _face_centers:
            out += f'{fc[0]:15.8f} {fc[1]:15.8f} {fc[2]:15.8f}\n'
        out += f'{n_faces}\n'
        for iface, face in enumerate(self._faces):
            if len(face) == 3:
                out += '4\n'
            elif len(face) == 4:
                out += '5\n'
            else:
                for ii, f in enumerate(face, start=1):
                    out += '4\n'
                    out += f'{len(_vertices) + iface}\n'
                    out += f'{f}\n'
                    out += f'{face[np.mod(ii, len(face))]}\n'
                    out += f'{len(_vertices) + iface}\n'
            if len(face) in [3, 4]:
                for face_vertex in face:
                    out += f'{face_vertex}\n'
                out += f'{face[0]}\n'
        pmeshes.append({'pmesh_string': out})
        return pmeshes