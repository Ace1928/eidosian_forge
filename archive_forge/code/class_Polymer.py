from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
class Polymer:
    """
    Generate polymer chain via Random walk. At each position there are
    a total of 5 possible moves(excluding the previous direction).
    """

    def __init__(self, start_monomer: Molecule, s_head: int, s_tail: int, monomer: Molecule, head: int, tail: int, end_monomer: Molecule, e_head: int, e_tail: int, n_units: int, link_distance: float=1.0, linear_chain: bool=False) -> None:
        """
        Args:
            start_monomer (Molecule): Starting molecule
            s_head (int): starting atom index of the start_monomer molecule
            s_tail (int): tail atom index of the start_monomer
            monomer (Molecule): The monomer
            head (int): index of the atom in the monomer that forms the head
            tail (int): tail atom index. monomers will be connected from
                tail to head
            end_monomer (Molecule): Terminal molecule
            e_head (int): starting atom index of the end_monomer molecule
            e_tail (int): tail atom index of the end_monomer
            n_units (int): number of monomer units excluding the start and
                terminal molecules
            link_distance (float): distance between consecutive monomers
            linear_chain (bool): linear or random walk polymer chain.
        """
        self.start = s_head
        self.end = s_tail
        self.monomer = monomer
        self.n_units = n_units
        self.link_distance = link_distance
        self.linear_chain = linear_chain
        start_monomer.translate_sites(range(len(start_monomer)), -monomer.cart_coords[s_head])
        monomer.translate_sites(range(len(monomer)), -monomer.cart_coords[head])
        end_monomer.translate_sites(range(len(end_monomer)), -monomer.cart_coords[e_head])
        self.mon_vector = monomer.cart_coords[tail] - monomer.cart_coords[head]
        self.moves = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1], 4: [-1, 0, 0], 5: [0, -1, 0], 6: [0, 0, -1]}
        self.prev_move = 1
        self.molecule = start_monomer.copy()
        self.length = 1
        self._create(self.monomer, self.mon_vector)
        self.n_units += 1
        end_mon_vector = end_monomer.cart_coords[e_tail] - end_monomer.cart_coords[e_head]
        self._create(end_monomer, end_mon_vector)
        self.molecule = Molecule.from_sites(self.molecule.sites)

    def _create(self, monomer: Molecule, mon_vector: ArrayLike) -> None:
        """
        create the polymer from the monomer.

        Args:
            monomer (Molecule)
            mon_vector (numpy.array): molecule vector that starts from the
                start atom index to the end atom index
        """
        while self.length != self.n_units - 1:
            if self.linear_chain:
                move_direction = np.array(mon_vector) / np.linalg.norm(mon_vector)
            else:
                move_direction = self._next_move_direction()
            self._add_monomer(monomer.copy(), mon_vector, move_direction)

    def _next_move_direction(self) -> np.ndarray:
        """Pick a move at random from the list of moves."""
        n_moves = len(self.moves)
        move = np.random.randint(1, n_moves + 1)
        while self.prev_move == (move + 3) % n_moves:
            move = np.random.randint(1, n_moves + 1)
        self.prev_move = move
        return np.array(self.moves[move])

    def _align_monomer(self, monomer: Molecule, mon_vector: ArrayLike, move_direction: ArrayLike) -> None:
        """
        rotate the monomer so that it is aligned along the move direction.

        Args:
            monomer (Molecule)
            mon_vector (numpy.array): molecule vector that starts from the
                start atom index to the end atom index
            move_direction (numpy.array): the direction of the polymer chain
                extension
        """
        axis = np.cross(mon_vector, move_direction)
        origin = monomer[self.start].coords
        angle = get_angle(mon_vector, move_direction)
        op = SymmOp.from_origin_axis_angle(origin, axis, angle)
        monomer.apply_operation(op)

    def _add_monomer(self, monomer: Molecule, mon_vector: ArrayLike, move_direction: ArrayLike) -> None:
        """
        extend the polymer molecule by adding a monomer along mon_vector direction.

        Args:
            monomer (Molecule): monomer molecule
            mon_vector (numpy.array): monomer vector that points from head to tail.
            move_direction (numpy.array): direction along which the monomer
                will be positioned
        """
        translate_by = self.molecule.cart_coords[self.end] + self.link_distance * move_direction
        monomer.translate_sites(range(len(monomer)), translate_by)
        if not self.linear_chain:
            self._align_monomer(monomer, mon_vector, move_direction)
        does_cross = False
        for idx, site in enumerate(monomer):
            try:
                self.molecule.append(site.specie, site.coords, properties=site.properties)
            except Exception:
                does_cross = True
                polymer_length = len(self.molecule)
                self.molecule.remove_sites(range(polymer_length - idx, polymer_length))
                break
        if not does_cross:
            self.length += 1
            self.end += len(self.monomer)