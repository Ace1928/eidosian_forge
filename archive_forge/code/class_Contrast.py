import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Contrast(FSCommand):
    """
    Compute surface-wise gray/white contrast

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import Contrast
    >>> contrast = Contrast()
    >>> contrast.inputs.subject_id = '10335'
    >>> contrast.inputs.hemisphere = 'lh'
    >>> contrast.inputs.white = 'lh.white' # doctest: +SKIP
    >>> contrast.inputs.thickness = 'lh.thickness' # doctest: +SKIP
    >>> contrast.inputs.annotation = '../label/lh.aparc.annot' # doctest: +SKIP
    >>> contrast.inputs.cortex = '../label/lh.cortex.label' # doctest: +SKIP
    >>> contrast.inputs.rawavg = '../mri/rawavg.mgz' # doctest: +SKIP
    >>> contrast.inputs.orig = '../mri/orig.mgz' # doctest: +SKIP
    >>> contrast.cmdline # doctest: +SKIP
    'pctsurfcon --lh-only --s 10335'
    """
    _cmd = 'pctsurfcon'
    input_spec = ContrastInputSpec
    output_spec = ContrastOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            hemi = self.inputs.hemisphere
            copy2subjdir(self, self.inputs.annotation, 'label', '{0}.aparc.annot'.format(hemi))
            copy2subjdir(self, self.inputs.cortex, 'label', '{0}.cortex.label'.format(hemi))
            copy2subjdir(self, self.inputs.white, 'surf', '{0}.white'.format(hemi))
            copy2subjdir(self, self.inputs.thickness, 'surf', '{0}.thickness'.format(hemi))
            copy2subjdir(self, self.inputs.orig, 'mri', 'orig.mgz')
            copy2subjdir(self, self.inputs.rawavg, 'mri', 'rawavg.mgz')
        createoutputdirs(self._list_outputs())
        return super(Contrast, self).run(**inputs)

    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id)
        outputs['out_contrast'] = os.path.join(subject_dir, 'surf', str(self.inputs.hemisphere) + '.w-g.pct.mgh')
        outputs['out_stats'] = os.path.join(subject_dir, 'stats', str(self.inputs.hemisphere) + '.w-g.pct.stats')
        outputs['out_log'] = os.path.join(subject_dir, 'scripts', 'pctsurfcon.log')
        return outputs