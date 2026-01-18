import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class AvScale(CommandLine):
    """Use FSL avscale command to extract info from mat file output of FLIRT

    Examples
    --------

    >>> avscale = AvScale()
    >>> avscale.inputs.mat_file = 'flirt.mat'
    >>> res = avscale.run()  # doctest: +SKIP


    """
    input_spec = AvScaleInputSpec
    output_spec = AvScaleOutputSpec
    _cmd = 'avscale'

    def _run_interface(self, runtime):
        runtime = super(AvScale, self)._run_interface(runtime)
        expr = re.compile('Rotation & Translation Matrix:\\n(?P<rot_tran_mat>[0-9\\. \\n-]+)[\\s\\n]*(Rotation Angles \\(x,y,z\\) \\[rads\\] = (?P<rot_angles>[0-9\\. -]+))?[\\s\\n]*(Translations \\(x,y,z\\) \\[mm\\] = (?P<translations>[0-9\\. -]+))?[\\s\\n]*Scales \\(x,y,z\\) = (?P<scales>[0-9\\. -]+)[\\s\\n]*Skews \\(xy,xz,yz\\) = (?P<skews>[0-9\\. -]+)[\\s\\n]*Average scaling = (?P<avg_scaling>[0-9\\.-]+)[\\s\\n]*Determinant = (?P<determinant>[0-9\\.-]+)[\\s\\n]*Left-Right orientation: (?P<lr_orientation>[A-Za-z]+)[\\s\\n]*Forward half transform =[\\s]*\\n(?P<fwd_half_xfm>[0-9\\. \\n-]+)[\\s\\n]*Backward half transform =[\\s]*\\n(?P<bwd_half_xfm>[0-9\\. \\n-]+)[\\s\\n]*')
        out = expr.search(runtime.stdout).groupdict()
        outputs = {}
        outputs['rotation_translation_matrix'] = [[float(v) for v in r.strip().split(' ')] for r in out['rot_tran_mat'].strip().split('\n')]
        outputs['scales'] = [float(s) for s in out['scales'].strip().split(' ')]
        outputs['skews'] = [float(s) for s in out['skews'].strip().split(' ')]
        outputs['average_scaling'] = float(out['avg_scaling'].strip())
        outputs['determinant'] = float(out['determinant'].strip())
        outputs['left_right_orientation_preserved'] = out['lr_orientation'].strip() == 'preserved'
        outputs['forward_half_transform'] = [[float(v) for v in r.strip().split(' ')] for r in out['fwd_half_xfm'].strip().split('\n')]
        outputs['backward_half_transform'] = [[float(v) for v in r.strip().split(' ')] for r in out['bwd_half_xfm'].strip().split('\n')]
        if self.inputs.all_param:
            outputs['rot_angles'] = [float(r) for r in out['rot_angles'].strip().split(' ')]
            outputs['translations'] = [float(r) for r in out['translations'].strip().split(' ')]
        setattr(self, '_results', outputs)
        return runtime

    def _list_outputs(self):
        return self._results