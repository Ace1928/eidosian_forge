from __future__ import annotations
import itertools
import os
from typing import TYPE_CHECKING
import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from monty.serialization import loadfn
from scipy.spatial import Delaunay
from pymatgen import vis
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
class AdsorbateSiteFinder:
    """This class finds adsorbate sites on slabs and generates adsorbate
    structures according to user-defined criteria.

    The algorithm for finding sites is essentially as follows:
        1. Determine "surface sites" by finding those within
            a height threshold along the miller index of the
            highest site
        2. Create a network of surface sites using the Delaunay
            triangulation of the surface sites
        3. Assign on-top, bridge, and hollow adsorption sites
            at the nodes, edges, and face centers of the Del.
            Triangulation
        4. Generate structures from a molecule positioned at
            these sites
    """

    def __init__(self, slab: Slab, selective_dynamics: bool=False, height: float=0.9, mi_vec: ArrayLike | None=None) -> None:
        """Create an AdsorbateSiteFinder object.

        Args:
            slab (Slab): slab object for which to find adsorbate sites
            selective_dynamics (bool): flag for whether to assign
                non-surface sites as fixed for selective dynamics
            height (float): height criteria for selection of surface sites
            mi_vec (3-D array-like): vector corresponding to the vector
                concurrent with the miller index, this enables use with
                slabs that have been reoriented, but the miller vector
                must be supplied manually
        """
        if mi_vec:
            self.mvec = mi_vec
        else:
            self.mvec = get_mi_vec(slab)
        slab = self.assign_site_properties(slab, height)
        if selective_dynamics:
            slab = self.assign_selective_dynamics(slab)
        self.slab = slab

    @classmethod
    def from_bulk_and_miller(cls, structure, miller_index, min_slab_size=8.0, min_vacuum_size=10.0, max_normal_search=None, center_slab=True, selective_dynamics=False, undercoord_threshold=0.09) -> Self:
        """This method constructs the adsorbate site finder from a bulk
        structure and a miller index, which allows the surface sites to be
        determined from the difference in bulk and slab coordination, as
        opposed to the height threshold.

        Args:
            structure (Structure): structure from which slab
                input to the ASF is constructed
            miller_index (3-tuple or list): miller index to be used
            min_slab_size (float): min slab size for slab generation
            min_vacuum_size (float): min vacuum size for slab generation
            max_normal_search (int): max normal search for slab generation
            center_slab (bool): whether to center slab in slab generation
            selective dynamics (bool): whether to assign surface sites
                to selective dynamics
            undercoord_threshold (float): threshold of "undercoordation"
                to use for the assignment of surface sites. Default is
                0.1, for which surface sites will be designated if they
                are 10% less coordinated than their bulk counterpart
        """
        vnn_bulk = VoronoiNN(tol=0.05)
        bulk_coords = [len(vnn_bulk.get_nn(structure, n)) for n in range(len(structure))]
        struct = structure.copy(site_properties={'bulk_coordinations': bulk_coords})
        slabs = generate_all_slabs(struct, max_index=max(miller_index), min_slab_size=min_slab_size, min_vacuum_size=min_vacuum_size, max_normal_search=max_normal_search, center_slab=center_slab)
        slab_dict = {slab.miller_index: slab for slab in slabs}
        if miller_index not in slab_dict:
            raise ValueError('Miller index not in slab dict')
        this_slab = slab_dict[miller_index]
        vnn_surface = VoronoiNN(tol=0.05, allow_pathological=True)
        surf_props, under_coords = ([], [])
        this_mi_vec = get_mi_vec(this_slab)
        mi_mags = [np.dot(this_mi_vec, site.coords) for site in this_slab]
        average_mi_mag = np.average(mi_mags)
        for n, site in enumerate(this_slab):
            bulk_coord = this_slab.site_properties['bulk_coordinations'][n]
            slab_coord = len(vnn_surface.get_nn(this_slab, n))
            mi_mag = np.dot(this_mi_vec, site.coords)
            under_coord = (bulk_coord - slab_coord) / bulk_coord
            under_coords += [under_coord]
            if under_coord > undercoord_threshold and mi_mag > average_mi_mag:
                surf_props += ['surface']
            else:
                surf_props += ['subsurface']
        new_site_properties = {'surface_properties': surf_props, 'undercoords': under_coords}
        new_slab = this_slab.copy(site_properties=new_site_properties)
        return cls(new_slab, selective_dynamics)

    def find_surface_sites_by_height(self, slab: Slab, height=0.9, xy_tol=0.05):
        """This method finds surface sites by determining which sites are
        within a threshold value in height from the topmost site in a list of
        sites.

        Args:
            slab (Slab): slab for which to find surface sites
            height (float): threshold in angstroms of distance from topmost
                site in slab along the slab c-vector to include in surface
                site determination
            xy_tol (float): if supplied, will remove any sites which are
                within a certain distance in the miller plane.

        Returns:
            list of sites selected to be within a threshold of the highest
        """
        m_projs = np.array([np.dot(site.coords, self.mvec) for site in slab])
        mask = m_projs - np.amax(m_projs) >= -height
        surf_sites = [slab.sites[n] for n in np.where(mask)[0]]
        if xy_tol:
            surf_sites = [s for h, s in zip(m_projs[mask], surf_sites)]
            surf_sites.reverse()
            unique_sites: list = []
            unique_perp_fracs: list = []
            for site in surf_sites:
                this_perp = site.coords - np.dot(site.coords, self.mvec)
                this_perp_frac = slab.lattice.get_fractional_coords(this_perp)
                if not in_coord_list_pbc(unique_perp_fracs, this_perp_frac):
                    unique_sites.append(site)
                    unique_perp_fracs.append(this_perp_frac)
            surf_sites = unique_sites
        return surf_sites

    def assign_site_properties(self, slab: Slab, height=0.9):
        """Assigns site properties."""
        if 'surface_properties' in slab.site_properties:
            return slab
        surf_sites = self.find_surface_sites_by_height(slab, height)
        surf_props = ['surface' if site in surf_sites else 'subsurface' for site in slab]
        return slab.copy(site_properties={'surface_properties': surf_props})

    def get_extended_surface_mesh(self, repeat=(5, 5, 1)):
        """Gets an extended surface mesh for to use for adsorption site finding
        by constructing supercell of surface sites.

        Args:
            repeat (3-tuple): repeat for getting extended surface mesh
        """
        surf_str = Structure.from_sites(self.surface_sites)
        surf_str.make_supercell(repeat)
        return surf_str

    @property
    def surface_sites(self):
        """Convenience method to return a list of surface sites."""
        return [site for site in self.slab if site.properties['surface_properties'] == 'surface']

    def subsurface_sites(self):
        """Convenience method to return list of subsurface sites."""
        return [site for site in self.slab if site.properties['surface_properties'] == 'subsurface']

    def find_adsorption_sites(self, distance=2.0, put_inside=True, symm_reduce=0.01, near_reduce=0.01, positions=('ontop', 'bridge', 'hollow'), no_obtuse_hollow=True):
        """Finds surface sites according to the above algorithm. Returns a list
        of corresponding Cartesian coordinates.

        Args:
            distance (float): distance from the coordinating ensemble
                of atoms along the miller index for the site (i. e.
                the distance from the slab itself)
            put_inside (bool): whether to put the site inside the cell
            symm_reduce (float): symm reduction threshold
            near_reduce (float): near reduction threshold
            positions (list): which positions to include in the site finding
                "ontop": sites on top of surface sites
                "bridge": sites at edges between surface sites in Delaunay
                    triangulation of surface sites in the miller plane
                "hollow": sites at centers of Delaunay triangulation faces
                "subsurface": subsurface positions projected into miller plane
            no_obtuse_hollow (bool): flag to indicate whether to include
                obtuse triangular ensembles in hollow sites
        """
        ads_sites = {k: [] for k in positions}
        if 'ontop' in positions:
            ads_sites['ontop'] = [s.coords for s in self.surface_sites]
        if 'subsurface' in positions:
            ref = self.slab.sites[np.argmax(self.slab.cart_coords[:, 2])]
            ss_sites = [self.mvec * np.dot(ref.coords - s.coords, self.mvec) + s.coords for s in self.subsurface_sites()]
            ads_sites['subsurface'] = ss_sites
        if 'bridge' in positions or 'hollow' in positions:
            mesh = self.get_extended_surface_mesh()
            symm_op = get_rot(self.slab)
            dt = Delaunay([symm_op.operate(m.coords)[:2] for m in mesh])
            for v in dt.simplices:
                if -1 not in v:
                    dots = []
                    for i_corner, i_opp in zip(range(3), ((1, 2), (0, 2), (0, 1))):
                        corner, opp = (v[i_corner], [v[o] for o in i_opp])
                        vecs = [mesh[d].coords - mesh[corner].coords for d in opp]
                        vecs = [vec / np.linalg.norm(vec) for vec in vecs]
                        dots.append(np.dot(*vecs))
                        if 'bridge' in positions:
                            ads_sites['bridge'].append(self.ensemble_center(mesh, opp))
                    obtuse = no_obtuse_hollow and (np.array(dots) < 1e-05).any()
                    if 'hollow' in positions and (not obtuse):
                        ads_sites['hollow'].append(self.ensemble_center(mesh, v))
        for key, sites in ads_sites.items():
            if key in ['bridge', 'hollow']:
                frac_coords = [self.slab.lattice.get_fractional_coords(ads_site) for ads_site in sites]
                frac_coords = [frac_coord for frac_coord in frac_coords if frac_coord[0] > 1 and frac_coord[0] < 4 and (frac_coord[1] > 1) and (frac_coord[1] < 4)]
                sites = [self.slab.lattice.get_cartesian_coords(frac_coord) for frac_coord in frac_coords]
            if near_reduce:
                sites = self.near_reduce(sites, threshold=near_reduce)
            if put_inside:
                sites = [put_coord_inside(self.slab.lattice, coord) for coord in sites]
            if symm_reduce:
                sites = self.symm_reduce(sites, threshold=symm_reduce)
            sites = [site + distance * np.asarray(self.mvec) for site in sites]
            ads_sites[key] = sites
        ads_sites['all'] = sum(ads_sites.values(), [])
        return ads_sites

    def symm_reduce(self, coords_set, threshold=1e-06):
        """Reduces the set of adsorbate sites by finding removing symmetrically
        equivalent duplicates.

        Args:
            coords_set: coordinate set in Cartesian coordinates
            threshold: tolerance for distance equivalence, used
                as input to in_coord_list_pbc for dupl. checking
        """
        surf_sg = SpacegroupAnalyzer(self.slab, 0.1)
        symm_ops = surf_sg.get_symmetry_operations()
        unique_coords = []
        coords_set = [self.slab.lattice.get_fractional_coords(coords) for coords in coords_set]
        for coords in coords_set:
            in_coord = False
            for op in symm_ops:
                if in_coord_list_pbc(unique_coords, op.operate(coords), atol=threshold):
                    in_coord = True
                    break
            if not in_coord:
                unique_coords += [coords]
        return [self.slab.lattice.get_cartesian_coords(coords) for coords in unique_coords]

    def near_reduce(self, coords_set, threshold=0.0001):
        """Prunes coordinate set for coordinates that are within threshold.

        Args:
            coords_set (Nx3 array-like): list or array of coordinates
            threshold (float): threshold value for distance
        """
        unique_coords = []
        coords_set = [self.slab.lattice.get_fractional_coords(coords) for coords in coords_set]
        for coord in coords_set:
            if not in_coord_list_pbc(unique_coords, coord, threshold):
                unique_coords += [coord]
        return [self.slab.lattice.get_cartesian_coords(coords) for coords in unique_coords]

    @classmethod
    def ensemble_center(cls, site_list, indices, cartesian=True):
        """Finds the center of an ensemble of sites selected from a list of
        sites. Helper method for the find_adsorption_sites algorithm.

        Args:
            site_list (list of sites): list of sites
            indices (list of ints): list of ints from which to select
                sites from site list
            cartesian (bool): whether to get average fractional or
                Cartesian coordinate
        """
        if cartesian:
            return np.average([site_list[idx].coords for idx in indices], axis=0)
        return np.average([site_list[idx].frac_coords for idx in indices], axis=0)

    def add_adsorbate(self, molecule: Molecule, ads_coord, repeat=None, translate=True, reorient=True):
        """Adds an adsorbate at a particular coordinate. Adsorbate represented
        by a Molecule object and is translated to (0, 0, 0) if translate is
        True, or positioned relative to the input adsorbate coordinate if
        translate is False.

        Args:
            molecule (Molecule): molecule object representing the adsorbate
            ads_coord (array): coordinate of adsorbate position
            repeat (3-tuple or list): input for making a supercell of slab
                prior to placing the adsorbate
            translate (bool): flag on whether to translate the molecule so
                that its CoM is at the origin prior to adding it to the surface
            reorient (bool): flag on whether to reorient the molecule to
                have its z-axis concurrent with miller index
        """
        molecule = molecule.copy()
        if translate:
            front_atoms = molecule.copy()
            front_atoms._sites = [s1 for s1 in molecule if s1.coords[2] == min((s2.coords[2] for s2 in molecule))]
            x, y, z = front_atoms.center_of_mass
            molecule.translate_sites(vector=[-x, -y, -z])
        if reorient:
            symm_op = get_rot(self.slab)
            molecule.apply_operation(symm_op.inverse)
        struct = self.slab.copy()
        if repeat:
            struct.make_supercell(repeat)
        if 'surface_properties' in struct.site_properties:
            molecule.add_site_property('surface_properties', ['adsorbate'] * len(molecule))
        if 'selective_dynamics' in struct.site_properties:
            molecule.add_site_property('selective_dynamics', [[True, True, True]] * len(molecule))
        for site in molecule:
            struct.append(site.specie, ads_coord + site.coords, coords_are_cartesian=True, properties=site.properties)
        return struct

    @classmethod
    def assign_selective_dynamics(cls, slab):
        """Helper function to assign selective dynamics site_properties based
        on surface, subsurface site properties.

        Args:
            slab (Slab): slab for which to assign selective dynamics
        """
        sd_list = []
        sd_list = [[False, False, False] if site.properties['surface_properties'] == 'subsurface' else [True, True, True] for site in slab]
        new_sp = slab.site_properties
        new_sp['selective_dynamics'] = sd_list
        return slab.copy(site_properties=new_sp)

    def generate_adsorption_structures(self, molecule, repeat=None, min_lw=5.0, translate=True, reorient=True, find_args=None):
        """Function that generates all adsorption structures for a given
        molecular adsorbate. Can take repeat argument or minimum length/width
        of precursor slab as an input.

        Args:
            molecule (Molecule): molecule corresponding to adsorbate
            repeat (3-tuple or list): repeat argument for supercell generation
            min_lw (float): minimum length and width of the slab, only used
                if repeat is None
            translate (bool): flag on whether to translate the molecule so
                that its CoM is at the origin prior to adding it to the surface
            reorient (bool): flag on whether or not to reorient adsorbate
                along the miller index
            find_args (dict): dictionary of arguments to be passed to the
                call to self.find_adsorption_sites, e.g. {"distance":2.0}
        """
        if repeat is None:
            xrep = np.ceil(min_lw / np.linalg.norm(self.slab.lattice.matrix[0]))
            yrep = np.ceil(min_lw / np.linalg.norm(self.slab.lattice.matrix[1]))
            repeat = [xrep, yrep, 1]
        structs = []
        find_args = find_args or {}
        for coords in self.find_adsorption_sites(**find_args)['all']:
            structs.append(self.add_adsorbate(molecule, coords, repeat=repeat, translate=translate, reorient=reorient))
        return structs

    def adsorb_both_surfaces(self, molecule, repeat=None, min_lw=5.0, translate=True, reorient=True, find_args=None):
        """Function that generates all adsorption structures for a given
        molecular adsorbate on both surfaces of a slab. This is useful for
        calculating surface energy where both surfaces need to be equivalent or
        if we want to calculate nonpolar systems.

        Args:
            molecule (Molecule): molecule corresponding to adsorbate
            repeat (3-tuple or list): repeat argument for supercell generation
            min_lw (float): minimum length and width of the slab, only used
                if repeat is None
            reorient (bool): flag on whether or not to reorient adsorbate
                along the miller index
            find_args (dict): dictionary of arguments to be passed to the
                call to self.find_adsorption_sites, e.g. {"distance":2.0}
        """
        find_args = find_args or {}
        ad_slabs = self.generate_adsorption_structures(molecule, repeat=repeat, min_lw=min_lw, translate=translate, reorient=reorient, find_args=find_args)
        new_ad_slabs = []
        for ad_slab in ad_slabs:
            _, adsorbates, indices = (False, [], [])
            for idx, site in enumerate(ad_slab):
                if site.surface_properties == 'adsorbate':
                    adsorbates.append(site)
                    indices.append(idx)
            ad_slab.remove_sites(indices)
            slab = ad_slab.copy()
            for adsorbate in adsorbates:
                p2 = ad_slab.get_symmetric_site(adsorbate.frac_coords)
                slab.append(adsorbate.specie, p2, properties={'surface_properties': 'adsorbate'})
                slab.append(adsorbate.specie, adsorbate.frac_coords, properties={'surface_properties': 'adsorbate'})
            new_ad_slabs.append(slab)
        return new_ad_slabs

    def generate_substitution_structures(self, atom, target_species=None, sub_both_sides=False, range_tol=0.01, dist_from_surf=0):
        """Function that performs substitution-type doping on the surface and
        returns all possible configurations where one dopant is substituted per
        surface. Can substitute one surface or both.

        Args:
            atom (str): atom corresponding to substitutional dopant
            sub_both_sides (bool): If true, substitute an equivalent
                site on the other surface
            target_species (list): List of specific species to substitute
            range_tol (float): Find viable substitution sites at a specific
                distance from the surface +- this tolerance
            dist_from_surf (float): Distance from the surface to find viable
                substitution sites, defaults to 0 to substitute at the surface
        """
        target_species = target_species or []
        sym_slab = SpacegroupAnalyzer(self.slab).get_symmetrized_structure()

        def substitute(site, idx):
            slab = self.slab.copy()
            props = self.slab.site_properties
            if sub_both_sides:
                eq_indices = next((indices for indices in sym_slab.equivalent_indices if idx in indices))
                for ii in eq_indices:
                    if f'{sym_slab[ii].frac_coords[2]:.6f}' != f'{site.frac_coords[2]:.6f}':
                        props['surface_properties'][ii] = 'substitute'
                        slab.replace(ii, atom)
                        break
            props['surface_properties'][idx] = 'substitute'
            slab.replace(idx, atom)
            slab.add_site_property('surface_properties', props['surface_properties'])
            return slab
        substituted_slabs = []
        sorted_sites = sorted(sym_slab, key=lambda site: site.frac_coords[2])
        if sorted_sites[0].surface_properties == 'surface':
            dist = sorted_sites[0].frac_coords[2] + dist_from_surf
        else:
            dist = sorted_sites[-1].frac_coords[2] - dist_from_surf
        for idx, site in enumerate(sym_slab):
            if dist - range_tol < site.frac_coords[2] < dist + range_tol and (target_species and site.species_string in target_species or not target_species):
                substituted_slabs.append(substitute(site, idx))
        matcher = StructureMatcher()
        return [s[0] for s in matcher.group_structures(substituted_slabs)]