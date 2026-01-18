from __future__ import annotations
import os
import tempfile
import warnings
from collections import namedtuple
from pathlib import Path
from shutil import which
from subprocess import Popen, TimeoutExpired
from monty.dev import requires
from pymatgen.core.structure import Structure
def _parse_sqs_path(path) -> Sqs:
    """Private function to parse mcsqs output directory
    Args:
        path: directory to perform parsing.

    Returns:
        tuple: Pymatgen structure SQS of the input structure, the mcsqs objective function,
            list of all SQS structures, and the directory where calculations are run
    """
    path = Path(path)
    detected_instances = len(list(path.glob('bestsqs*[0-9]*.out')))
    with Popen('str2cif < bestsqs.out > bestsqs.cif', shell=True, cwd=path) as p:
        p.communicate()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        best_sqs = Structure.from_file(path / 'bestsqs.out')
    with open(path / 'bestcorr.out') as file:
        lines = file.readlines()
    objective_function_str = lines[-1].split('=')[-1].strip()
    objective_function: float | str
    objective_function = float(objective_function_str) if objective_function_str != 'Perfect_match' else 'Perfect_match'
    all_sqs = []
    for idx in range(detected_instances):
        sqs_out = f'bestsqs{idx + 1}.out'
        sqs_cif = f'bestsqs{idx + 1}.cif'
        corr_out = f'bestcorr{idx + 1}.out'
        with Popen(f'str2cif < {sqs_out} > {sqs_cif}', shell=True, cwd=path) as p:
            p.communicate()
        sqs = Structure.from_file(path / sqs_out)
        with open(path / corr_out) as file:
            lines = file.readlines()
        objective_function_str = lines[-1].split('=')[-1].strip()
        obj: float | str
        obj = float(objective_function_str) if objective_function_str != 'Perfect_match' else 'Perfect_match'
        all_sqs.append({'structure': sqs, 'objective_function': obj})
    clusters = _parse_clusters(path / 'clusters.out')
    return Sqs(bestsqs=best_sqs, objective_function=objective_function, allsqs=all_sqs, directory=str(path.resolve()), clusters=clusters)