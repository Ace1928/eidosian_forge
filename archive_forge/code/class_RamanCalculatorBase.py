import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
class RamanCalculatorBase(IOContext):

    def __init__(self, atoms, *args, name='raman', exext='.alpha', txt='-', verbose=False, comm=world, **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exext: string
          Extension for excitation filenames
        txt:
          Output stream
        verbose:
          Verbosity level of output
        comm:
          Communicator, default world
        """
        kwargs['name'] = name
        self.exname = kwargs.pop('exname', name)
        super().__init__(atoms, *args, **kwargs)
        self.exext = exext
        self.txt = self.openfile(txt, comm)
        self.verbose = verbose
        self.comm = comm