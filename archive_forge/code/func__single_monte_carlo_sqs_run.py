from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def _single_monte_carlo_sqs_run(self):
    """Run a single Monte Carlo SQS search with Icet."""
    cluster_space = self._get_cluster_space()
    sqs_structure = generate_sqs(cluster_space=cluster_space, max_size=self.scaling, target_concentrations=self.target_concentrations, **self.sqs_kwargs)
    return {'structure': sqs_structure, 'objective_function': self.get_icet_sqs_obj(sqs_structure, cluster_space=cluster_space)}