from pathlib import Path
from ase.io import write
from ase.io.elk import ElkReader
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.abc import GetOutputsMixin
Construct ELK calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        