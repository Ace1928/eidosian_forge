from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
class ASENeighborList(NeighborList):

    def __init__(self, compute_args, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, debug):
        super().__init__(neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, debug)
        self.neigh = {}
        compute_args.set_callback(kimpy.compute_callback_name.GetNeighborList, self.get_neigh, self.neigh)

    @staticmethod
    def get_neigh(data, cutoffs, neighbor_list_index, particle_number):
        """Retrieves the neighbors of each atom using ASE's native neighbor
        list library
        """
        number_of_particles = data['num_particles']
        if particle_number >= number_of_particles or particle_number < 0:
            return (np.array([]), 1)
        neighbors = data['neighbors'][neighbor_list_index][particle_number]
        return (neighbors, 0)

    def build(self, orig_atoms):
        """Build the ASE neighbor list and return an Atoms object with
        all of the neighbors added.  First a neighbor list is created
        from ase.neighbor_list, having only information about the
        neighbors of the original atoms.  If neighbors of padding atoms
        are required, they are calculated using information from the
        first neighbor list.
        """
        syms = orig_atoms.get_chemical_symbols()
        orig_num_atoms = len(orig_atoms)
        orig_pos = orig_atoms.get_positions()
        new_atoms = orig_atoms.copy()
        neigh_list = defaultdict(list)
        neigh_dists = defaultdict(list)
        padding_image_of = []
        padding_shifts = []
        neigh_indices_i, neigh_indices_j, relative_pos, neigh_cell_offsets, dists = neighbor_list('ijDSd', orig_atoms, self.influence_dist)
        used = dict()
        for neigh_i, neigh_j, rel_pos, offset, dist in zip(neigh_indices_i, neigh_indices_j, relative_pos, neigh_cell_offsets, dists):
            wrapped_pos = orig_pos[neigh_i] + rel_pos
            shift = tuple(offset)
            uniq_index = (neigh_j,) + shift
            if shift == (0, 0, 0):
                neigh_list[neigh_i].append(neigh_j)
                neigh_dists[neigh_i].append(dist)
                if uniq_index not in used:
                    used[uniq_index] = neigh_j
            else:
                if uniq_index not in used:
                    used[uniq_index] = len(new_atoms)
                    new_atoms.append(Atom(syms[neigh_j], position=wrapped_pos))
                    padding_image_of.append(neigh_j)
                    padding_shifts.append(offset)
                neigh_list[neigh_i].append(used[uniq_index])
                neigh_dists[neigh_i].append(dist)
        neighbor_list_size = orig_num_atoms
        if self.padding_need_neigh:
            neighbor_list_size = len(new_atoms)
            inv_used = dict(((v, k) for k, v in used.items()))
            for k, neigh in enumerate(padding_image_of):
                shift = padding_shifts[k]
                for orig_neigh, orig_dist in zip(neigh_list[neigh], neigh_dists[neigh]):
                    orig_shift = inv_used[orig_neigh][1:]
                    total_shift = orig_shift + shift
                    if orig_neigh <= orig_num_atoms - 1:
                        orig_neigh_image = orig_neigh
                    else:
                        orig_neigh_image = padding_image_of[orig_neigh - orig_num_atoms]
                    uniq_index = (orig_neigh_image,) + tuple(total_shift)
                    if uniq_index in used:
                        neigh_list[k + orig_num_atoms].append(used[uniq_index])
                        neigh_dists[k + orig_num_atoms].append(orig_dist)
        neigh_lists = []
        for cut in self.cutoffs:
            neigh_list = [np.array(neigh_list[k], dtype=np.intc)[neigh_dists[k] <= cut] for k in range(neighbor_list_size)]
            neigh_lists.append(neigh_list)
        self.padding_image_of = padding_image_of
        self.neigh['neighbors'] = neigh_lists
        self.neigh['num_particles'] = neighbor_list_size
        return new_atoms

    def update(self, orig_atoms, species_map):
        """Create the neighbor list along with the other required
        parameters (which are stored as instance attributes). The
        required parameters are:

            - num_particles
            - coords
            - particle_contributing
            - species_code

        Note that the KIM API requires a neighbor list that has indices
        corresponding to each atom.
        """
        self.num_contributing_particles = len(orig_atoms)
        new_atoms = self.build(orig_atoms)
        num_atoms = len(new_atoms)
        num_padding = num_atoms - self.num_contributing_particles
        self.num_particles = [num_atoms]
        self.coords = new_atoms.get_positions()
        indices_mask = [1] * self.num_contributing_particles + [0] * num_padding
        self.particle_contributing = indices_mask
        try:
            self.species_code = [species_map[s] for s in new_atoms.get_chemical_symbols()]
        except KeyError as e:
            raise RuntimeError('Species not supported by KIM model; {}'.format(str(e)))
        self.last_update_positions = orig_atoms.get_positions()
        if self.debug:
            print('Debug: called update_ase_neigh')
            print()