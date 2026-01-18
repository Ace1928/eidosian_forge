from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging
class CiftiSmoothInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='The input CIFTI file')
    sigma_surf = traits.Float(mandatory=True, argstr='%s', position=1, desc='the sigma for the gaussian surface smoothing kernel, in mm')
    sigma_vol = traits.Float(mandatory=True, argstr='%s', position=2, desc='the sigma for the gaussian volume smoothing kernel, in mm')
    direction = traits.Enum('ROW', 'COLUMN', mandatory=True, argstr='%s', position=3, desc='which dimension to smooth along, ROW or COLUMN')
    out_file = File(name_source=['in_file'], name_template='smoothed_%s.nii', keep_extension=True, argstr='%s', position=4, desc='The output CIFTI')
    left_surf = File(exists=True, mandatory=True, position=5, argstr='-left-surface %s', desc='Specify the left surface to use')
    left_corrected_areas = File(exists=True, position=6, argstr='-left-corrected-areas %s', desc='vertex areas (as a metric) to use instead of computing them from the left surface.')
    right_surf = File(exists=True, mandatory=True, position=7, argstr='-right-surface %s', desc='Specify the right surface to use')
    right_corrected_areas = File(exists=True, position=8, argstr='-right-corrected-areas %s', desc='vertex areas (as a metric) to use instead of computing them from the right surface')
    cerebellum_surf = File(exists=True, position=9, argstr='-cerebellum-surface %s', desc='specify the cerebellum surface to use')
    cerebellum_corrected_areas = File(exists=True, position=10, requires=['cerebellum_surf'], argstr='cerebellum-corrected-areas %s', desc='vertex areas (as a metric) to use instead of computing them from the cerebellum surface')
    cifti_roi = File(exists=True, position=11, argstr='-cifti-roi %s', desc='CIFTI file for ROI smoothing')
    fix_zeros_vol = traits.Bool(position=12, argstr='-fix-zeros-volume', desc='treat values of zero in the volume as missing data')
    fix_zeros_surf = traits.Bool(position=13, argstr='-fix-zeros-surface', desc='treat values of zero on the surface as missing data')
    merged_volume = traits.Bool(position=14, argstr='-merged-volume', desc='smooth across subcortical structure boundaries')