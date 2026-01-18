from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
class Nwchem2Fiesta(MSONable):
    """
    To run NWCHEM2FIESTA inside python:

    If nwchem.nw is the input, nwchem.out the output, and structure.movecs the
    "movecs" file, the syntax to run NWCHEM2FIESTA is: NWCHEM2FIESTA
    nwchem.nw  nwchem.nwout  structure.movecs > log_n2f
    """

    def __init__(self, folder, filename='nwchem', log_file='log_n2f'):
        """
        folder: where are stored the nwchem
        filename: name of nwchem files read by NWCHEM2FIESTA (filename.nw, filename.nwout and filename.movecs)
        logfile: logfile of NWCHEM2FIESTA.

        the run method launches NWCHEM2FIESTA
        """
        self.folder = folder
        self.filename = filename
        self.log_file = log_file
        self._NWCHEM2FIESTA_cmd = 'NWCHEM2FIESTA'
        self._nwcheminput_fn = filename + '.nw'
        self._nwchemoutput_fn = filename + '.nwout'
        self._nwchemmovecs_fn = filename + '.movecs'

    def run(self):
        """Performs actual NWCHEM2FIESTA run."""
        init_folder = os.getcwd()
        os.chdir(self.folder)
        with zopen(self.log_file, mode='w') as fout:
            subprocess.call([self._NWCHEM2FIESTA_cmd, self._nwcheminput_fn, self._nwchemoutput_fn, self._nwchemmovecs_fn], stdout=fout)
        os.chdir(init_folder)

    def as_dict(self):
        """MSONable dict"""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'filename': self.filename, 'folder': self.folder}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Nwchem2Fiesta
        """
        return cls(folder=dct['folder'], filename=dct['filename'])