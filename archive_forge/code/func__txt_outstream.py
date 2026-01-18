import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
@contextmanager
def _txt_outstream(self):
    """Custom function for opening a text output stream. Uses self.txt
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes
        the new stream again when exiting.

        Examples:
        # Pass a string
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'

        # Use an existing stream
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout
        """
    txt = self.txt
    open_and_close = False
    if txt is None:
        out = subprocess.DEVNULL
    elif isinstance(txt, str):
        if txt == '-':
            out = None
        else:
            txt = self._indir(txt)
            open_and_close = True
    elif hasattr(txt, 'write'):
        out = txt
    else:
        raise RuntimeError('txt should either be a stringor an I/O stream, got {}'.format(txt))
    try:
        if open_and_close:
            out = open(txt, 'w')
        yield out
    finally:
        if open_and_close:
            out.close()