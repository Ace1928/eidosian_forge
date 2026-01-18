import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def _update_data_groups(self, params_old, params_update):
    """updates data groups in the control file"""
    grps = []
    for p in list(params_update.keys()):
        if self.parameter_group[p] is not None:
            grps.append(self.parameter_group[p])
    dgs = {}
    for g in grps:
        dgs[g] = {}
        for p in self.parameter_key:
            if g == self.parameter_group[p]:
                if self.parameter_group[p] == self.parameter_key[p]:
                    if p in list(params_update.keys()):
                        val = params_update[p]
                        pmap = list(self.parameter_mapping.keys())
                        if val is not None and p in pmap:
                            fun = self.parameter_mapping[p]['to_control']
                            val = fun(params_update[p])
                        dgs[g] = val
                else:
                    if p in list(params_old.keys()):
                        val = params_old[p]
                        pmap = list(self.parameter_mapping.keys())
                        if val is not None and p in pmap:
                            fun = self.parameter_mapping[p]['to_control']
                            val = fun(params_old[p])
                        dgs[g][self.parameter_key[p]] = val
                    if p in list(params_update.keys()):
                        val = params_update[p]
                        pmap = list(self.parameter_mapping.keys())
                        if val is not None and p in pmap:
                            fun = self.parameter_mapping[p]['to_control']
                            val = fun(params_update[p])
                        dgs[g][self.parameter_key[p]] = val
    for g in dgs:
        delete_data_group(g)
        if isinstance(dgs[g], dict):
            string = ''
            for key in list(dgs[g].keys()):
                if dgs[g][key] is None:
                    continue
                elif isinstance(dgs[g][key], bool):
                    if dgs[g][key]:
                        string += ' ' + key
                else:
                    string += ' ' + key + '=' + str(dgs[g][key])
            add_data_group(g, string=string)
        elif isinstance(dgs[g], bool):
            if dgs[g]:
                add_data_group(g, string='')
        else:
            add_data_group(g, string=str(dgs[g]))
    self._set_post_define()