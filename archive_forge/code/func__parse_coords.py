from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
@staticmethod
def _parse_coords(coord_lines):
    """Helper method to parse coordinates."""
    paras = {}
    var_pattern = re.compile('^([A-Za-z]+\\S*)[\\s=,]+([\\d\\-\\.]+)$')
    for line in coord_lines:
        m = var_pattern.match(line.strip())
        if m:
            paras[m.group(1).strip('=')] = float(m.group(2))
    species = []
    coords = []
    zmode = False
    for line in coord_lines:
        line = line.strip()
        if not line:
            break
        if not zmode and GaussianInput._xyz_patt.match(line):
            m = GaussianInput._xyz_patt.match(line)
            species.append(m.group(1))
            tokens = re.split('[,\\s]+', line.strip())
            if len(tokens) > 4:
                coords.append([float(i) for i in tokens[2:5]])
            else:
                coords.append([float(i) for i in tokens[1:4]])
        elif GaussianInput._zmat_patt.match(line):
            zmode = True
            tokens = re.split('[,\\s]+', line.strip())
            species.append(tokens[0])
            tokens.pop(0)
            if len(tokens) == 0:
                coords.append(np.array([0, 0, 0]))
            else:
                nn = []
                parameters = []
                while len(tokens) > 1:
                    ind = tokens.pop(0)
                    data = tokens.pop(0)
                    try:
                        nn.append(int(ind))
                    except ValueError:
                        nn.append(species.index(ind) + 1)
                    try:
                        val = float(data)
                        parameters.append(val)
                    except ValueError:
                        if data.startswith('-'):
                            parameters.append(-paras[data[1:]])
                        else:
                            parameters.append(paras[data])
                if len(nn) == 1:
                    coords.append(np.array([0, 0, parameters[0]]))
                elif len(nn) == 2:
                    coords1 = coords[nn[0] - 1]
                    coords2 = coords[nn[1] - 1]
                    bl = parameters[0]
                    angle = parameters[1]
                    axis = [0, 1, 0]
                    op = SymmOp.from_origin_axis_angle(coords1, axis, angle)
                    coord = op.operate(coords2)
                    vec = coord - coords1
                    coord = vec * bl / np.linalg.norm(vec) + coords1
                    coords.append(coord)
                elif len(nn) == 3:
                    coords1 = coords[nn[0] - 1]
                    coords2 = coords[nn[1] - 1]
                    coords3 = coords[nn[2] - 1]
                    bl = parameters[0]
                    angle = parameters[1]
                    dih = parameters[2]
                    v1 = coords3 - coords2
                    v2 = coords1 - coords2
                    axis = np.cross(v1, v2)
                    op = SymmOp.from_origin_axis_angle(coords1, axis, angle)
                    coord = op.operate(coords2)
                    v1 = coord - coords1
                    v2 = coords1 - coords2
                    v3 = np.cross(v1, v2)
                    adj = get_angle(v3, axis)
                    axis = coords1 - coords2
                    op = SymmOp.from_origin_axis_angle(coords1, axis, dih - adj)
                    coord = op.operate(coord)
                    vec = coord - coords1
                    coord = vec * bl / np.linalg.norm(vec) + coords1
                    coords.append(coord)

    def _parse_species(sp_str):
        """
            The species specification can take many forms. E.g.,
            simple integers representing atomic numbers ("8"),
            actual species string ("C") or a labelled species ("C1").
            Sometimes, the species string is also not properly capitalized,
            e.g, ("c1"). This method should take care of these known formats.
            """
        try:
            return int(sp_str)
        except ValueError:
            sp = re.sub('\\d', '', sp_str)
            return sp.capitalize()
    species = [_parse_species(sp) for sp in species]
    return Molecule(species, coords)