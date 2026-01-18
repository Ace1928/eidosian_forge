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
class MultiStructuresInteractorStyle(StructureInteractorStyle):
    """Interactor for MultiStructureVis."""

    def __init__(self, parent) -> None:
        StructureInteractorStyle.__init__(self, parent=parent)

    def keyPressEvent(self, obj, event):
        parent = obj.GetCurrentRenderer().parent
        sym = parent.iren.GetKeySym()
        if sym == 'n':
            if parent.istruct == len(parent.structures) - 1:
                parent.display_warning('LAST STRUCTURE')
                parent.ren_win.Render()
            else:
                parent.istruct += 1
                parent.current_structure = parent.structures[parent.istruct]
                parent.set_structure(parent.current_structure, reset_camera=False, to_unit_cell=False)
                parent.erase_warning()
                parent.ren_win.Render()
        elif sym == 'p':
            if parent.istruct == 0:
                parent.display_warning('FIRST STRUCTURE')
                parent.ren_win.Render()
            else:
                parent.istruct -= 1
                parent.current_structure = parent.structures[parent.istruct]
                parent.set_structure(parent.current_structure, reset_camera=False, to_unit_cell=False)
                parent.erase_warning()
                parent.ren_win.Render()
        elif sym == 'm':
            parent.istruct = 0
            parent.current_structure = parent.structures[parent.istruct]
            parent.set_structure(parent.current_structure, reset_camera=False, to_unit_cell=False)
            parent.erase_warning()
            parent.ren_win.Render()
            nloops = parent.animated_movie_options['number_of_loops']
            tstep = parent.animated_movie_options['time_between_frames']
            tloops = parent.animated_movie_options['time_between_loops']
            if parent.animated_movie_options['looping_type'] == 'restart':
                loop_istructs = range(len(parent.structures))
            elif parent.animated_movie_options['looping_type'] == 'palindrome':
                loop_istructs = range(len(parent.structures)) + range(len(parent.structures) - 2, -1, -1)
            else:
                raise ValueError('"looping_type" should be "restart" or "palindrome"')
            for iloop in range(nloops):
                for istruct in loop_istructs:
                    time.sleep(tstep)
                    parent.istruct = istruct
                    parent.current_structure = parent.structures[parent.istruct]
                    parent.set_structure(parent.current_structure, reset_camera=False, to_unit_cell=False)
                    parent.display_info(f'Animated movie : structure {istruct + 1}/{len(parent.structures)} (loop {iloop + 1}/{nloops})')
                    parent.ren_win.Render()
                time.sleep(tloops)
            parent.erase_info()
            parent.display_info('Ended animated movie ...')
            parent.ren_win.Render()
        StructureInteractorStyle.keyPressEvent(self, obj, event)