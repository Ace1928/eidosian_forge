from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
class POVRAY:
    material_styles_dict = dict(simple='finish {phong 0.7}', pale='finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }', intermediate='finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}', vmd='finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }', jmol='finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}', ase2='finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}', ase3='finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}', glass='finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}', glass2='finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}')

    def __init__(self, cell, cell_vertices, positions, diameters, colors, image_width, image_height, constraints=tuple(), isosurfaces=[], display=False, pause=True, transparent=True, canvas_width=None, canvas_height=None, camera_dist=50.0, image_plane=None, camera_type='orthographic', point_lights=[], area_light=[(2.0, 3.0, 40.0), 'White', 0.7, 0.7, 3, 3], background='White', textures=None, transmittances=None, depth_cueing=False, cue_density=0.005, celllinewidth=0.05, bondlinewidth=0.1, bondatoms=[], exportconstraints=False):
        """
        # x, y is the image plane, z is *out* of the screen
        cell: ase.cell
            cell object
        cell_vertices: 2-d numpy array
            contains the 8 vertices of the cell, each with three coordinates
        positions: 2-d numpy array
            number of atoms length array with three coordinates for positions
        diameters: 1-d numpy array
            diameter of atoms (in order with positions)
        colors: list of str
            colors of atoms (in order with positions)
        image_width: float
            image width in pixels
        image_height: float
            image height in pixels
        constraints: Atoms.constraints
            constraints to be visualized
        isosurfaces: list of POVRAYIsosurface
            composite object to write/render POVRAY isosurfaces
        display: bool
            display while rendering
        pause: bool
            pause when done rendering (only if display)
        transparent: bool
            make background transparent
        canvas_width: int
            width of canvas in pixels
        canvas_height: int
            height of canvas in pixels
        camera_dist: float
            distance from camera to front atom
        image_plane: float
            distance from front atom to image plane
        camera_type: str
            if 'orthographic' perspective, ultra_wide_angle
        point_lights: list of 2-element sequences
            like [[loc1, color1], [loc2, color2],...]
        area_light: 3-element sequence of location (3-tuple), color (str),
                   width (float), height (float),
                   Nlamps_x (int), Nlamps_y (int)
            example [(2., 3., 40.), 'White', .7, .7, 3, 3]
        background: str
            color specification, e.g., 'White'
        textures: list of str
            length of atoms list of texture names
        transmittances: list of floats
            length of atoms list of transmittances of the atoms
        depth_cueing: bool
            whether or not to use depth cueing a.k.a. fog
            use with care - adjust the camera_distance to be closer
        cue_density: float
            if there is depth_cueing, how dense is it (how dense is the fog)
        celllinewidth: float
            radius of the cylinders representing the cell (Ang.)
        bondlinewidth: float
            radius of the cylinders representing bonds (Ang.)
        bondatoms: list of lists (polymorphic)
            [[atom1, atom2], ... ] pairs of bonding atoms
             For bond order > 1 = [[atom1, atom2, offset,
                                    bond_order, bond_offset],
                                   ... ]
             bond_order: 1, 2, 3 for single, double,
                          and triple bond
             bond_offset: vector for shifting bonds from
                           original position. Coordinates are
                           in Angstrom unit.
        exportconstraints: bool
            honour FixAtoms and mark?"""
        self.area_light = area_light
        self.background = background
        self.bondatoms = bondatoms
        self.bondlinewidth = bondlinewidth
        self.camera_dist = camera_dist
        self.camera_type = camera_type
        self.celllinewidth = celllinewidth
        self.cue_density = cue_density
        self.depth_cueing = depth_cueing
        self.display = display
        self.exportconstraints = exportconstraints
        self.isosurfaces = isosurfaces
        self.pause = pause
        self.point_lights = point_lights
        self.textures = textures
        self.transmittances = transmittances
        self.transparent = transparent
        self.image_width = image_width
        self.image_height = image_height
        self.colors = colors
        self.cell = cell
        self.diameters = diameters
        z0 = positions[:, 2].max()
        self.offset = (image_width / 2, image_height / 2, z0)
        self.positions = positions - self.offset
        if cell_vertices is not None:
            self.cell_vertices = cell_vertices - self.offset
            self.cell_vertices.shape = (2, 2, 2, 3)
        else:
            self.cell_vertices = None
        ratio = float(self.image_width) / self.image_height
        if canvas_width is None:
            if canvas_height is None:
                self.canvas_width = min(self.image_width * 15, 640)
                self.canvas_height = min(self.image_height * 15, 640)
            else:
                self.canvas_width = canvas_height * ratio
                self.canvas_height = canvas_height
        elif canvas_height is None:
            self.canvas_width = canvas_width
            self.canvas_height = self.canvas_width / ratio
        else:
            raise RuntimeError("Can't set *both* width and height!")
        if image_plane is None:
            if self.camera_type == 'orthographic':
                self.image_plane = 1 - self.camera_dist
            else:
                self.image_plane = 0
        self.image_plane += self.camera_dist
        self.constrainatoms = []
        for c in constraints:
            if isinstance(c, FixAtoms):
                for n, i in enumerate(c.index):
                    self.constrainatoms += [i]

    @classmethod
    def from_PlottingVariables(cls, pvars, **kwargs):
        cell = pvars.cell
        cell_vertices = pvars.cell_vertices
        if 'colors' in kwargs.keys():
            colors = kwargs.pop('colors')
        else:
            colors = pvars.colors
        diameters = pvars.d
        image_height = pvars.h
        image_width = pvars.w
        positions = pvars.positions
        constraints = pvars.constraints
        return cls(cell=cell, cell_vertices=cell_vertices, colors=colors, constraints=constraints, diameters=diameters, image_height=image_height, image_width=image_width, positions=positions, **kwargs)

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls.from_plotting_variables(PlottingVariables(atoms, scale=1.0), **kwargs)

    def write_ini(self, path):
        """Write ini file."""
        ini_str = f'Input_File_Name={path.with_suffix('.pov').name}\nOutput_to_File=True\nOutput_File_Type=N\nOutput_Alpha={('on' if self.transparent else 'off')}\n; if you adjust Height, and width, you must preserve the ratio\n; Width / Height = {self.canvas_width / self.canvas_height:f}\nWidth={self.canvas_width}\nHeight={self.canvas_height}\nAntialias=True\nAntialias_Threshold=0.1\nDisplay={self.display}\nPause_When_Done={self.pause}\nVerbose=False\n'
        with open(path, 'w') as fd:
            fd.write(ini_str)
        return path

    def write_pov(self, path):
        """Write pov file."""
        point_lights = '\n'.join((f'light_source {{{pa(loc)} {pc(rgb)}}}' for loc, rgb in self.point_lights))
        area_light = ''
        if self.area_light is not None:
            loc, color, width, height, nx, ny = self.area_light
            area_light += f'\nlight_source {{{pa(loc)} {pc(color)}\n  area_light <{width:.2f}, 0, 0>, <0, {height:.2f}, 0>, {nx:n}, {ny:n}\n  adaptive 1 jitter}}'
        fog = ''
        if self.depth_cueing and self.cue_density >= 0.0001:
            if self.cue_density > 10000.0:
                dist = 0.0001
            else:
                dist = 1.0 / self.cue_density
            fog += f'fog {{fog_type 1 distance {dist:.4f} color {pc(self.background)}}}'
        mat_style_keys = (f'#declare {k} = {v}' for k, v in self.material_styles_dict.items())
        mat_style_keys = '\n'.join(mat_style_keys)
        cell_vertices = ''
        if self.cell_vertices is not None:
            for c in range(3):
                for j in ([0, 0], [1, 0], [1, 1], [0, 1]):
                    p1 = self.cell_vertices[tuple(j[:c]) + (0,) + tuple(j[c:])]
                    p2 = self.cell_vertices[tuple(j[:c]) + (1,) + tuple(j[c:])]
                    distance = np.linalg.norm(p2 - p1)
                    if distance < 1e-12:
                        continue
                    cell_vertices += f'cylinder {{{pa(p1)}, {pa(p2)}, Rcell pigment {{Black}}}}\n'
            cell_vertices = cell_vertices.strip('\n')
        a = 0
        atoms = ''
        for loc, dia, col in zip(self.positions, self.diameters, self.colors):
            tex = 'ase3'
            trans = 0.0
            if self.textures is not None:
                tex = self.textures[a]
            if self.transmittances is not None:
                trans = self.transmittances[a]
            atoms += f'atom({pa(loc)}, {dia / 2.0:.2f}, {pc(col)}, {trans}, {tex}) // #{a:n}\n'
            a += 1
        atoms = atoms.strip('\n')
        bondatoms = ''
        for pair in self.bondatoms:
            if len(pair) == 2:
                a, b = pair
                offset = (0, 0, 0)
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 3:
                a, b, offset = pair
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 4:
                a, b, offset, bond_order = pair
                bond_offset = (self.bondlinewidth, self.bondlinewidth, 0)
            elif len(pair) > 4:
                a, b, offset, bond_order, bond_offset = pair
            else:
                raise RuntimeError('Each list in bondatom must have at least 2 entries. Error at %s' % pair)
            if len(offset) != 3:
                raise ValueError('offset must have 3 elements. Error at %s' % pair)
            if len(bond_offset) != 3:
                raise ValueError('bond_offset must have 3 elements. Error at %s' % pair)
            if bond_order not in [0, 1, 2, 3]:
                raise ValueError('bond_order must be either 0, 1, 2, or 3. Error at %s' % pair)
            if bond_order > 1 and np.linalg.norm(bond_offset) > 1e-09:
                tmp_atoms = Atoms('H3')
                tmp_atoms.set_cell(self.cell)
                tmp_atoms.set_positions([self.positions[a], self.positions[b], self.positions[b] + np.array(bond_offset)])
                tmp_atoms.center()
                tmp_atoms.set_angle(0, 1, 2, 90)
                bond_offset = tmp_atoms[2].position - tmp_atoms[1].position
            R = np.dot(offset, self.cell)
            mida = 0.5 * (self.positions[a] + self.positions[b] + R)
            midb = 0.5 * (self.positions[a] + self.positions[b] - R)
            if self.textures is not None:
                texa = self.textures[a]
                texb = self.textures[b]
            else:
                texa = texb = 'ase3'
            if self.transmittances is not None:
                transa = self.transmittances[a]
                transb = self.transmittances[b]
            else:
                transa = transb = 0.0
            posa = self.positions[a]
            posb = self.positions[b]
            cola = self.colors[a]
            colb = self.colors[b]
            if bond_order == 1:
                draw_tuples = ((posa, mida, cola, transa, texa), (posb, midb, colb, transb, texb))
            elif bond_order == 2:
                bs = [x / 2 for x in bond_offset]
                draw_tuples = ((posa - bs, mida - bs, cola, transa, texa), (posb - bs, midb - bs, colb, transb, texb), (posa + bs, mida + bs, cola, transa, texa), (posb + bs, midb + bs, colb, transb, texb))
            elif bond_order == 3:
                bs = bond_offset
                draw_tuples = ((posa, mida, cola, transa, texa), (posb, midb, colb, transb, texb), (posa + bs, mida + bs, cola, transa, texa), (posb + bs, midb + bs, colb, transb, texb), (posa - bs, mida - bs, cola, transa, texa), (posb - bs, midb - bs, colb, transb, texb))
            bondatoms += ''.join((f'cylinder {{{pa(p)}, {pa(m)}, Rbond texture{{pigment {{color {pc(c)} transmit {tr}}} finish{{{tx}}}}}}}\n' for p, m, c, tr, tx in draw_tuples))
        bondatoms = bondatoms.strip('\n')
        constraints = ''
        if self.exportconstraints:
            for a in self.constrainatoms:
                dia = self.diameters[a]
                loc = self.positions[a]
                trans = 0.0
                if self.transmittances is not None:
                    trans = self.transmittances[a]
                constraints += f'constrain({pa(loc)}, {dia / 2.0:.2f}, Black, {trans}, {tex}) // #{a:n} \n'
        constraints = constraints.strip('\n')
        pov = f'#include "colors.inc"\n#include "finish.inc"\n\nglobal_settings {{assumed_gamma 1 max_trace_level 6}}\nbackground {{{pc(self.background)}{(' transmit 1.0' if self.transparent else '')}}}\ncamera {{{self.camera_type}\n  right -{self.image_width:.2f}*x up {self.image_height:.2f}*y\n  direction {self.image_plane:.2f}*z\n  location <0,0,{self.camera_dist:.2f}> look_at <0,0,0>}}\n{point_lights}\n{(area_light if area_light != '' else '// no area light')}\n{(fog if fog != '' else '// no fog')}\n{mat_style_keys}\n#declare Rcell = {self.celllinewidth:.3f};\n#declare Rbond = {self.bondlinewidth:.3f};\n\n#macro atom(LOC, R, COL, TRANS, FIN)\n  sphere{{LOC, R texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n#end\n#macro constrain(LOC, R, COL, TRANS FIN)\nunion{{torus{{R, Rcell rotate 45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n     torus{{R, Rcell rotate -45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n     translate LOC}}\n#end\n\n{(cell_vertices if cell_vertices != '' else '// no cell vertices')}\n{atoms}\n{bondatoms}\n{(constraints if constraints != '' else '// no constraints')}\n'
        with open(path, 'w') as fd:
            fd.write(pov)
        return path

    def write(self, pov_path):
        pov_path = require_pov(pov_path)
        ini_path = pov_path.with_suffix('.ini')
        self.write_ini(ini_path)
        self.write_pov(pov_path)
        if self.isosurfaces is not None:
            with open(pov_path, 'a') as fd:
                for iso in self.isosurfaces:
                    fd.write(iso.format_mesh())
        return POVRAYInputs(ini_path)