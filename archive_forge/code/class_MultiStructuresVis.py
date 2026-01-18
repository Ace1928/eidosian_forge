from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
class MultiStructuresVis(StructureVis):
    """Visualization for multiple structures."""
    DEFAULT_ANIMATED_MOVIE_OPTIONS = dict(time_between_frames=0.1, looping_type='restart', number_of_loops=1, time_between_loops=1.0)

    def __init__(self, element_color_mapping=None, show_unit_cell=True, show_bonds=False, show_polyhedron=False, poly_radii_tol_factor=0.5, excluded_bonding_elements=None, animated_movie_options=DEFAULT_ANIMATED_MOVIE_OPTIONS):
        """
        Args:
            element_color_mapping: Optional color mapping for the elements,
                as a dict of {symbol: rgb tuple}. For example, {"Fe": (255,
                123,0), ....} If None is specified, a default based on
                Jmol's color scheme is used.
            show_unit_cell: Set to False to not show the unit cell
                boundaries. Defaults to True.
            show_bonds: Set to True to show bonds. Defaults to True.
            show_polyhedron: Set to True to show polyhedrons. Defaults to
                False.
            poly_radii_tol_factor: The polyhedron and bonding code uses the
                ionic radii of the elements or species to determine if two
                atoms are bonded. This specifies a tolerance scaling factor
                such that atoms which are (1 + poly_radii_tol_factor) * sum
                of ionic radii apart are still considered as bonded.
            excluded_bonding_elements: List of atom types to exclude from
                bonding determination. Defaults to an empty list. Useful
                when trying to visualize a certain atom type in the
                framework (e.g., Li in a Li-ion battery cathode material).
            animated_movie_options (): Used for moving.
        """
        super().__init__(element_color_mapping=element_color_mapping, show_unit_cell=show_unit_cell, show_bonds=show_bonds, show_polyhedron=show_polyhedron, poly_radii_tol_factor=poly_radii_tol_factor, excluded_bonding_elements=excluded_bonding_elements)
        self.warning_txt_actor = vtk.vtkActor2D()
        self.info_txt_actor = vtk.vtkActor2D()
        self.structures = None
        style = MultiStructuresInteractorStyle(self)
        self.iren.SetInteractorStyle(style)
        self.istruct = 0
        self.current_structure = None
        self.set_animated_movie_options(animated_movie_options=animated_movie_options)

    def set_structures(self, structures: Sequence[Structure], tags=None):
        """
        Add list of structures to the visualizer.

        Args:
            structures (list[Structures]): structures to be visualized.
            tags (): List of tags.
        """
        self.structures = structures
        self.istruct = 0
        self.current_structure = self.structures[self.istruct]
        self.tags = tags if tags is not None else []
        self.all_radii = []
        self.all_vis_radii = []
        for struct in self.structures:
            struct_radii = []
            struct_vis_radii = []
            for site in struct:
                radius = 0
                for specie, occu in site.species.items():
                    radius += occu * (specie.ionic_radius if isinstance(specie, Species) and specie.ionic_radius else specie.average_ionic_radius)
                    vis_radius = 0.2 + 0.002 * radius
                struct_radii.append(radius)
                struct_vis_radii.append(vis_radius)
            self.all_radii.append(struct_radii)
            self.all_vis_radii.append(struct_vis_radii)
        self.set_structure(self.current_structure, reset_camera=True, to_unit_cell=False)

    def set_structure(self, structure: Structure, reset_camera=True, to_unit_cell=False):
        """
        Add a structure to the visualizer.

        Args:
            structure: structure to visualize
            reset_camera: Set to True to reset the camera to a default
                determined based on the structure.
            to_unit_cell: Whether or not to fall back sites into the unit cell.
        """
        super().set_structure(structure=structure, reset_camera=reset_camera, to_unit_cell=to_unit_cell)
        self.apply_tags()

    def apply_tags(self):
        """Apply tags."""
        tags = {}
        for tag in self.tags:
            istruct = tag.get('istruct', 'all')
            if istruct not in ('all', self.istruct):
                continue
            site_index = tag['site_index']
            color = tag.get('color', [0.5, 0.5, 0.5])
            opacity = tag.get('opacity', 0.5)
            if site_index == 'unit_cell_all':
                struct_radii = self.all_vis_radii[self.istruct]
                for isite, _site in enumerate(self.current_structure):
                    vis_radius = 1.5 * tag.get('radius', struct_radii[isite])
                    tags[isite, (0, 0, 0)] = {'radius': vis_radius, 'color': color, 'opacity': opacity}
                continue
            cell_index = tag['cell_index']
            if 'radius' in tag:
                vis_radius = tag['radius']
            elif 'radius_factor' in tag:
                vis_radius = tag['radius_factor'] * self.all_vis_radii[self.istruct][site_index]
            else:
                vis_radius = 1.5 * self.all_vis_radii[self.istruct][site_index]
            tags[site_index, cell_index] = {'radius': vis_radius, 'color': color, 'opacity': opacity}
        for site_and_cell_index, tag_style in tags.items():
            isite, cell_index = site_and_cell_index
            site = self.current_structure[isite]
            if cell_index == (0, 0, 0):
                coords = site.coords
            else:
                fcoords = site.frac_coords + np.array(cell_index)
                site_image = PeriodicSite(site.species, fcoords, self.current_structure.lattice, to_unit_cell=False, coords_are_cartesian=False, properties=site.properties)
                self.add_site(site_image)
                coords = site_image.coords
            vis_radius = tag_style['radius']
            color = tag_style['color']
            opacity = tag_style['opacity']
            self.add_partial_sphere(coords=coords, radius=vis_radius, color=color, start=0, end=360, opacity=opacity)

    def set_animated_movie_options(self, animated_movie_options=None):
        """
        Args:
            animated_movie_options ():
        """
        if animated_movie_options is None:
            self.animated_movie_options = self.DEFAULT_ANIMATED_MOVIE_OPTIONS.copy()
        else:
            self.animated_movie_options = self.DEFAULT_ANIMATED_MOVIE_OPTIONS.copy()
            for key in animated_movie_options:
                if key not in self.DEFAULT_ANIMATED_MOVIE_OPTIONS:
                    raise ValueError('Wrong option for animated movie')
            self.animated_movie_options.update(animated_movie_options)

    def display_help(self):
        """Display the help for various keyboard shortcuts."""
        helptxt = ['h : Toggle help', 'A/a, B/b or C/c : Increase/decrease cell by one a, b or c unit vector', '# : Toggle showing of polyhedrons', '-: Toggle showing of bonds', 'r : Reset camera direction', f'[/]: Decrease or increase poly_radii_tol_factor by 0.05. Value = {self.poly_radii_tol_factor}', 'Up/Down: Rotate view along Up direction by 90 clockwise/anticlockwise', 'Left/right: Rotate view along camera direction by 90 clockwise/anticlockwise', 's: Save view to image.png', 'o: Orthogonalize structure', 'n: Move to next structure', 'p: Move to previous structure', 'm: Animated movie of the structures']
        self.helptxt_mapper.SetInput('\n'.join(helptxt))
        self.helptxt_actor.SetPosition(10, 10)
        self.helptxt_actor.VisibilityOn()

    def display_warning(self, warning):
        """
        Args:
            warning (str): Warning.
        """
        self.warning_txt_mapper = vtk.vtkTextMapper()
        tprops = self.warning_txt_mapper.GetTextProperty()
        tprops.SetFontSize(14)
        tprops.SetFontFamilyToTimes()
        tprops.SetColor(1, 0, 0)
        tprops.BoldOn()
        tprops.SetJustificationToRight()
        self.warning_txt = f'WARNING : {warning}'
        self.warning_txt_actor = vtk.vtkActor2D()
        self.warning_txt_actor.VisibilityOn()
        self.warning_txt_actor.SetMapper(self.warning_txt_mapper)
        self.ren.AddActor(self.warning_txt_actor)
        self.warning_txt_mapper.SetInput(self.warning_txt)
        winsize = self.ren_win.GetSize()
        self.warning_txt_actor.SetPosition(winsize[0] - 10, 10)
        self.warning_txt_actor.VisibilityOn()

    def erase_warning(self):
        """Remove warnings."""
        self.warning_txt_actor.VisibilityOff()

    def display_info(self, info):
        """
        Args:
            info (str): Information.
        """
        self.info_txt_mapper = vtk.vtkTextMapper()
        tprops = self.info_txt_mapper.GetTextProperty()
        tprops.SetFontSize(14)
        tprops.SetFontFamilyToTimes()
        tprops.SetColor(0, 0, 1)
        tprops.BoldOn()
        tprops.SetVerticalJustificationToTop()
        self.info_txt = f'INFO : {info}'
        self.info_txt_actor = vtk.vtkActor2D()
        self.info_txt_actor.VisibilityOn()
        self.info_txt_actor.SetMapper(self.info_txt_mapper)
        self.ren.AddActor(self.info_txt_actor)
        self.info_txt_mapper.SetInput(self.info_txt)
        winsize = self.ren_win.GetSize()
        self.info_txt_actor.SetPosition(10, winsize[1] - 10)
        self.info_txt_actor.VisibilityOn()

    def erase_info(self):
        """Erase all info."""
        self.info_txt_actor.VisibilityOff()