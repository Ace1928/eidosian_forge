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
class StructureInteractorStyle(TrackballCamera):
    """A custom interactor style for visualizing structures."""

    def __init__(self, parent):
        self.parent = parent
        self.AddObserver('LeftButtonPressEvent', self.leftButtonPressEvent)
        self.AddObserver('MouseMoveEvent', self.mouseMoveEvent)
        self.AddObserver('LeftButtonReleaseEvent', self.leftButtonReleaseEvent)
        self.AddObserver('KeyPressEvent', self.keyPressEvent)

    def leftButtonPressEvent(self, obj, event):
        self.mouse_motion = 0
        self.OnLeftButtonDown()

    def mouseMoveEvent(self, obj, event):
        self.mouse_motion = 1
        self.OnMouseMove()

    def leftButtonReleaseEvent(self, obj, event):
        ren = obj.GetCurrentRenderer()
        iren = ren.GetRenderWindow().GetInteractor()
        if self.mouse_motion == 0:
            pos = iren.GetEventPosition()
            iren.GetPicker().Pick(pos[0], pos[1], 0, ren)
        self.OnLeftButtonUp()

    def keyPressEvent(self, obj, _event):
        parent = obj.GetCurrentRenderer().parent
        sym = parent.iren.GetKeySym()
        if sym in 'ABCabc':
            if sym == 'A':
                parent.supercell[0][0] += 1
            elif sym == 'B':
                parent.supercell[1][1] += 1
            elif sym == 'C':
                parent.supercell[2][2] += 1
            elif sym == 'a':
                parent.supercell[0][0] = max(parent.supercell[0][0] - 1, 1)
            elif sym == 'b':
                parent.supercell[1][1] = max(parent.supercell[1][1] - 1, 1)
            elif sym == 'c':
                parent.supercell[2][2] = max(parent.supercell[2][2] - 1, 1)
            parent.redraw()
        elif sym == 'numbersign':
            parent.show_polyhedron = not parent.show_polyhedron
            parent.redraw()
        elif sym == 'minus':
            parent.show_bonds = not parent.show_bonds
            parent.redraw()
        elif sym == 'bracketleft':
            parent.poly_radii_tol_factor -= 0.05 if parent.poly_radii_tol_factor > 0 else 0
            parent.redraw()
        elif sym == 'bracketright':
            parent.poly_radii_tol_factor += 0.05
            parent.redraw()
        elif sym == 'h':
            parent.show_help = not parent.show_help
            parent.redraw()
        elif sym == 'r':
            parent.redraw(True)
        elif sym == 's':
            parent.write_image('image.png')
        elif sym == 'Up':
            parent.rotate_view(1, 90)
        elif sym == 'Down':
            parent.rotate_view(1, -90)
        elif sym == 'Left':
            parent.rotate_view(0, -90)
        elif sym == 'Right':
            parent.rotate_view(0, 90)
        elif sym == 'o':
            parent.orthogonalize_structure()
            parent.redraw()
        self.OnKeyPress()