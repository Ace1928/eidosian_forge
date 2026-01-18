import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
class Dcm2niix(CommandLine):
    """Uses Chris Rorden's dcm2niix to convert dicom files

    Examples
    ========

    >>> from nipype.interfaces.dcm2nii import Dcm2niix
    >>> converter = Dcm2niix()
    >>> converter.inputs.source_dir = 'dicomdir'
    >>> converter.inputs.compression = 5
    >>> converter.inputs.output_dir = 'ds005'
    >>> converter.cmdline
    'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n dicomdir'
    >>> converter.run() # doctest: +SKIP

    In the example below, we note that the current version of dcm2niix
    converts any files in the directory containing the files in the list. We
    also do not support nested filenames with this option. **Thus all files
    must have a common root directory.**

    >>> converter = Dcm2niix()
    >>> converter.inputs.source_names = ['functional_1.dcm', 'functional_2.dcm']
    >>> converter.inputs.compression = 5
    >>> converter.inputs.output_dir = 'ds005'
    >>> converter.cmdline
    'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n .'
    >>> converter.run() # doctest: +SKIP
    """
    input_spec = Dcm2niixInputSpec
    output_spec = Dcm2niixOutputSpec
    _cmd = 'dcm2niix'

    @property
    def version(self):
        return Info.version()

    def _format_arg(self, opt, spec, val):
        bools = ['bids_format', 'merge_imgs', 'single_file', 'verbose', 'crop', 'has_private', 'anon_bids', 'ignore_deriv', 'philips_float', 'to_nrrd']
        if opt in bools:
            spec = deepcopy(spec)
            if val:
                spec.argstr += ' y'
            else:
                spec.argstr += ' n'
                val = True
        if opt == 'source_names':
            return spec.argstr % (os.path.dirname(val[0]) or '.')
        return super(Dcm2niix, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime):
        runtime = super(Dcm2niix, self)._run_interface(runtime, correct_return_codes=(0, 1))
        self._parse_files(self._parse_stdout(runtime.stdout))
        return runtime

    def _parse_stdout(self, stdout):
        filenames = []
        for line in stdout.split('\n'):
            if line.startswith('Convert '):
                fname = str(re.search('\\S+/\\S+', line).group(0))
                filenames.append(os.path.abspath(fname))
        return filenames

    def _parse_files(self, filenames):
        outfiles, bvals, bvecs, bids = ([], [], [], [])
        outtypes = ['.bval', '.bvec', '.json', '.txt']
        if self.inputs.to_nrrd:
            outtypes += ['.nrrd', '.nhdr', '.raw.gz']
        else:
            outtypes += ['.nii', '.nii.gz']
        for filename in filenames:
            for fl in search_files(filename, outtypes):
                if fl.endswith('.nii') or fl.endswith('.gz') or fl.endswith('.nrrd') or fl.endswith('.nhdr'):
                    outfiles.append(fl)
                elif fl.endswith('.bval'):
                    bvals.append(fl)
                elif fl.endswith('.bvec'):
                    bvecs.append(fl)
                elif fl.endswith('.json') or fl.endswith('.txt'):
                    bids.append(fl)
        self.output_files = outfiles
        self.bvecs = bvecs
        self.bvals = bvals
        self.bids = bids

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['converted_files'] = self.output_files
        outputs['bvecs'] = self.bvecs
        outputs['bvals'] = self.bvals
        outputs['bids'] = self.bids
        return outputs