from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def _prepare_sources(mname, sources, bdir):
    extended_sources = sources.copy()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    for source in sources:
        if Path(source).exists() and Path(source).is_file():
            shutil.copy(source, bdir)
    generated_sources = [Path(f'{mname}module.c'), Path(f'{mname}-f2pywrappers2.f90'), Path(f'{mname}-f2pywrappers.f')]
    bdir = Path(bdir)
    for generated_source in generated_sources:
        if generated_source.exists():
            shutil.copy(generated_source, bdir / generated_source.name)
            extended_sources.append(generated_source.name)
            generated_source.unlink()
    extended_sources = [Path(source).name for source in extended_sources if not Path(source).suffix == '.pyf']
    return extended_sources