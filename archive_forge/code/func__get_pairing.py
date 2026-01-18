import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
from ase.ga.offspring_creator import OffspringCreator
def _get_pairing(self, a1, a2, cutting_point, cutting_normal, cell):
    """Creates a child from two parents using the given cut.

        Returns None if the generated structure does not contain
        a large enough fraction of each parent (see self.minfrac).

        Does not check whether atoms are too close.

        Assumes the 'slab' parts have been removed from the parent
        structures and that these have been checked for equal
        lengths, stoichiometries, and tags (if self.use_tags).

        Parameters:

        cutting_normal: int or (1x3) array

        cutting_point: (1x3) array
            In fractional coordinates

        cell: (3x3) array
            The unit cell for the child structure
        """
    symbols = a1.get_chemical_symbols()
    tags = a1.get_tags() if self.use_tags else np.arange(len(a1))
    p1, p2, sym = ([], [], [])
    for i in np.unique(tags):
        indices = np.where(tags == i)[0]
        s = ''.join([symbols[j] for j in indices])
        sym.append(s)
        for i, (a, p) in enumerate(zip([a1, a2], [p1, p2])):
            c = a.get_cell()
            cop = np.mean(a.positions[indices], axis=0)
            cut_p = np.dot(cutting_point, c)
            if isinstance(cutting_normal, int):
                vecs = [c[j] for j in range(3) if j != cutting_normal]
                cut_n = np.cross(vecs[0], vecs[1])
            else:
                cut_n = np.dot(cutting_normal, c)
            d = np.dot(cop - cut_p, cut_n)
            spos = a.get_scaled_positions()[indices]
            scop = np.mean(spos, axis=0)
            p.append(Positions(spos, scop, s, d, i))
    all_points = p1 + p2
    unique_sym = np.unique(sym)
    types = {s: sym.count(s) for s in unique_sym}
    all_points.sort(key=lambda x: x.symbols, reverse=True)
    unique_sym.sort()
    use_total = dict()
    for s in unique_sym:
        used = []
        not_used = []
        for i in reversed(range(len(all_points))):
            if all_points[i].symbols != s:
                break
            if all_points[i].to_use():
                used.append(all_points.pop(i))
            else:
                not_used.append(all_points.pop(i))
        assert len(used) + len(not_used) == types[s] * 2
        while len(used) < types[s]:
            index = self.rng.randint(len(not_used))
            used.append(not_used.pop(index))
        while len(used) > types[s]:
            index = self.rng.randint(len(used))
            not_used.append(used.pop(index))
        use_total[s] = used
    n_tot = sum([len(ll) for ll in use_total.values()])
    assert n_tot == len(sym)
    count1, count2, N = (0, 0, len(a1))
    for x in use_total.values():
        count1 += sum([y.origin == 0 for y in x])
        count2 += sum([y.origin == 1 for y in x])
    nmin = 1 if self.minfrac is None else int(round(self.minfrac * N))
    if count1 < nmin or count2 < nmin:
        return None
    newpos = []
    pbc = a1.get_pbc()
    for s in sym:
        p = use_total[s].pop()
        c = a1.get_cell() if p.origin == 0 else a2.get_cell()
        pos = np.dot(p.scaled_positions, c)
        cop = np.dot(p.cop, c)
        vectors, lengths = find_mic(pos - cop, c, pbc)
        newcop = np.dot(p.cop, cell)
        pos = newcop + vectors
        for row in pos:
            newpos.append(row)
    newpos = np.reshape(newpos, (N, 3))
    num = a1.get_atomic_numbers()
    child = Atoms(numbers=num, positions=newpos, pbc=pbc, cell=cell, tags=tags)
    child.wrap()
    return child