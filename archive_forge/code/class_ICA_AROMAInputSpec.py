from ..base import (
import os
class ICA_AROMAInputSpec(CommandLineInputSpec):
    feat_dir = Directory(exists=True, mandatory=True, argstr='-feat %s', xor=['in_file', 'mat_file', 'fnirt_warp_file', 'motion_parameters'], desc='If a feat directory exists and temporal filtering has not been run yet, ICA_AROMA can use the files in this directory.')
    in_file = File(exists=True, mandatory=True, argstr='-i %s', xor=['feat_dir'], desc='volume to be denoised')
    out_dir = Directory('out', usedefault=True, mandatory=True, argstr='-o %s', desc='output directory')
    mask = File(exists=True, argstr='-m %s', xor=['feat_dir'], desc='path/name volume mask')
    dim = traits.Int(argstr='-dim %d', desc='Dimensionality reduction when running MELODIC (default is automatic estimation)')
    TR = traits.Float(argstr='-tr %.3f', desc='TR in seconds. If this is not specified the TR will be extracted from the header of the fMRI nifti file.')
    melodic_dir = Directory(exists=True, argstr='-meldir %s', desc='path to MELODIC directory if MELODIC has already been run')
    mat_file = File(exists=True, argstr='-affmat %s', xor=['feat_dir'], desc='path/name of the mat-file describing the affine registration (e.g. FSL FLIRT) of the functional data to structural space (.mat file)')
    fnirt_warp_file = File(exists=True, argstr='-warp %s', xor=['feat_dir'], desc='File name of the warp-file describing the non-linear registration (e.g. FSL FNIRT) of the structural data to MNI152 space (.nii.gz)')
    motion_parameters = File(exists=True, mandatory=True, argstr='-mc %s', xor=['feat_dir'], desc='motion parameters file')
    denoise_type = traits.Enum('nonaggr', 'aggr', 'both', 'no', usedefault=True, mandatory=True, argstr='-den %s', desc='Type of denoising strategy:\n-no: only classification, no denoising\n-nonaggr (default): non-aggresssive denoising, i.e. partial component regression\n-aggr: aggressive denoising, i.e. full component regression\n-both: both aggressive and non-aggressive denoising (two outputs)')