import os
from ....base import (
class fibertrackInputSpec(CommandLineInputSpec):
    input_tensor_file = File(desc='Tensor Image', exists=True, argstr='--input_tensor_file %s')
    input_roi_file = File(desc='The filename of the image which contains the labels used for seeding and constraining the algorithm.', exists=True, argstr='--input_roi_file %s')
    output_fiber_file = traits.Either(traits.Bool, File(), hash_files=False, desc='The filename for the fiber file produced by the algorithm. This file must end in a .fib or .vtk extension for ITK spatial object and vtkPolyData formats respectively.', argstr='--output_fiber_file %s')
    source_label = traits.Int(desc='The label of voxels in the labelfile to use for seeding tractography. One tract is seeded from the center of each voxel with this label', argstr='--source_label %d')
    target_label = traits.Int(desc='The label of voxels in the labelfile used to constrain tractography. Tracts that do not pass through a voxel with this label are rejected. Set this keep all tracts.', argstr='--target_label %d')
    forbidden_label = traits.Int(desc='Forbidden label', argstr='--forbidden_label %d')
    whole_brain = traits.Bool(desc='If this option is enabled all voxels in the image are used to seed tractography. When this option is enabled both source and target labels function as target labels', argstr='--whole_brain ')
    max_angle = traits.Float(desc='Maximum angle of change in radians', argstr='--max_angle %f')
    step_size = traits.Float(desc='Step size in mm for the tracking algorithm', argstr='--step_size %f')
    min_fa = traits.Float(desc='The minimum FA threshold to continue tractography', argstr='--min_fa %f')
    force = traits.Bool(desc='Ignore sanity checks.', argstr='--force ')
    verbose = traits.Bool(desc='produce verbose output', argstr='--verbose ')
    really_verbose = traits.Bool(desc='Follow detail of fiber tracking algorithm', argstr='--really_verbose ')