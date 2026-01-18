from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
def _create_input(self):
    structure = self.structure
    mcbs = self.mc_box_size
    equil_timesteps = self.equil_timesteps
    mc_timesteps = self.mc_timesteps
    mat_name = self.mat_name
    input_script = [f'material:unit-cell-file={mat_name}.ucf']
    input_script += [f'material:file={mat_name}.mat']
    input_script += ['create:periodic-boundaries-x', 'create:periodic-boundaries-y', 'create:periodic-boundaries-z']
    abc = structure.lattice.abc
    ucx, ucy, ucz = (abc[0], abc[1], abc[2])
    input_script += [f'dimensions:unit-cell-size-x = {ucx:.10f} !A']
    input_script += [f'dimensions:unit-cell-size-y = {ucy:.10f} !A']
    input_script += [f'dimensions:unit-cell-size-z = {ucz:.10f} !A']
    input_script += [f'dimensions:system-size-x = {mcbs:.1f} !nm', f'dimensions:system-size-y = {mcbs:.1f} !nm', f'dimensions:system-size-z = {mcbs:.1f} !nm']
    input_script += ['sim:integrator = monte-carlo', 'sim:program = curie-temperature']
    input_script += [f'sim:equilibration-time-steps = {equil_timesteps}', f'sim:loop-time-steps = {mc_timesteps}', 'sim:time-steps-increment = 1']
    start_t = self.user_input_settings.get('start_t', 0)
    end_t = self.user_input_settings.get('end_t', 1500)
    temp_increment = self.user_input_settings.get('temp_increment', 25)
    input_script += [f'sim:minimum-temperature = {start_t}', f'sim:maximum-temperature = {end_t}', f'sim:temperature-increment = {temp_increment}']
    input_script += ['output:temperature', 'output:mean-magnetisation-length', 'output:material-mean-magnetisation-length', 'output:mean-susceptibility']
    input_script = '\n'.join(input_script)
    with open('input', mode='w') as file:
        file.write(input_script)