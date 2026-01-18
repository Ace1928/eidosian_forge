import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class UnpackSDICOMDirInputSpec(FSTraitedSpec):
    source_dir = Directory(exists=True, argstr='-src %s', mandatory=True, desc='directory with the DICOM files')
    output_dir = Directory(argstr='-targ %s', desc='top directory into which the files will be unpacked')
    run_info = traits.Tuple(traits.Int, traits.Str, traits.Str, traits.Str, mandatory=True, argstr='-run %d %s %s %s', xor=('run_info', 'config', 'seq_config'), desc='runno subdir format name : spec unpacking rules on cmdline')
    config = File(exists=True, argstr='-cfg %s', mandatory=True, xor=('run_info', 'config', 'seq_config'), desc='specify unpacking rules in file')
    seq_config = File(exists=True, argstr='-seqcfg %s', mandatory=True, xor=('run_info', 'config', 'seq_config'), desc='specify unpacking rules based on sequence')
    dir_structure = traits.Enum('fsfast', 'generic', argstr='-%s', desc='unpack to specified directory structures')
    no_info_dump = traits.Bool(argstr='-noinfodump', desc='do not create infodump file')
    scan_only = File(exists=True, argstr='-scanonly %s', desc='only scan the directory and put result in file')
    log_file = File(exists=True, argstr='-log %s', desc='explicitly set log file')
    spm_zeropad = traits.Int(argstr='-nspmzeropad %d', desc='set frame number zero padding width for SPM')
    no_unpack_err = traits.Bool(argstr='-no-unpackerr', desc='do not try to unpack runs with errors')