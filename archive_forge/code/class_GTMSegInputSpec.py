import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class GTMSegInputSpec(FSTraitedSpec):
    subject_id = traits.String(argstr='--s %s', desc='subject id', mandatory=True)
    xcerseg = traits.Bool(argstr='--xcerseg', desc='run xcerebralseg on this subject to create apas+head.mgz')
    out_file = File('gtmseg.mgz', argstr='--o %s', desc='output volume relative to subject/mri', usedefault=True)
    upsampling_factor = traits.Int(argstr='--usf %i', desc='upsampling factor (default is 2)')
    subsegwm = traits.Bool(argstr='--subsegwm', default=True, desc='subsegment WM into lobes (default)')
    keep_hypo = traits.Bool(argstr='--keep-hypo', desc='do not relabel hypointensities as WM when subsegmenting WM')
    keep_cc = traits.Bool(argstr='--keep-cc', desc='do not relabel corpus callosum as WM')
    dmax = traits.Float(argstr='--dmax %f', desc='distance threshold to use when subsegmenting WM (default is 5)')
    ctx_annot = traits.Tuple(traits.String, traits.Int, traits.Int, argstr='--ctx-annot %s %i %i', desc='annot lhbase rhbase : annotation to use for cortical segmentation (default is aparc 1000 2000)')
    wm_annot = traits.Tuple(traits.String, traits.Int, traits.Int, argstr='--wm-annot %s %i %i', desc='annot lhbase rhbase : annotation to use for WM segmentation (with --subsegwm, default is lobes 3200 4200)')
    output_upsampling_factor = traits.Int(argstr='--output-usf %i', desc='set output USF different than USF, mostly for debugging')
    head = traits.String(argstr='--head %s', desc='use headseg instead of apas+head.mgz')
    subseg_cblum_wm = traits.Bool(argstr='--subseg-cblum-wm', desc='subsegment cerebellum WM into core and gyri')
    no_pons = traits.Bool(argstr='--no-pons', desc='do not add pons segmentation when doing ---xcerseg')
    no_vermis = traits.Bool(argstr='--no-vermis', desc='do not add vermis segmentation when doing ---xcerseg')
    colortable = File(exists=True, argstr='--ctab %s', desc='colortable')
    no_seg_stats = traits.Bool(argstr='--no-seg-stats', desc='do not compute segmentation stats')