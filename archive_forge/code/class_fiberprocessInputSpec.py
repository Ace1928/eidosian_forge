import os
from ....base import (
class fiberprocessInputSpec(CommandLineInputSpec):
    fiber_file = File(desc='DTI fiber file', exists=True, argstr='--fiber_file %s')
    fiber_output = traits.Either(traits.Bool, File(), hash_files=False, desc='Output fiber file. May be warped or updated with new data depending on other options used.', argstr='--fiber_output %s')
    tensor_volume = File(desc='Interpolate tensor values from the given field', exists=True, argstr='--tensor_volume %s')
    h_field = File(desc='HField for warp and statistics lookup. If this option is used tensor-volume must also be specified.', exists=True, argstr='--h_field %s')
    displacement_field = File(desc='Displacement Field for warp and statistics lookup.  If this option is used tensor-volume must also be specified.', exists=True, argstr='--displacement_field %s')
    saveProperties = traits.Bool(desc='save the tensor property as scalar data into the vtk (only works for vtk fiber files). ', argstr='--saveProperties ')
    no_warp = traits.Bool(desc='Do not warp the geometry of the tensors only obtain the new statistics.', argstr='--no_warp ')
    fiber_radius = traits.Float(desc='set radius of all fibers to this value', argstr='--fiber_radius %f')
    index_space = traits.Bool(desc='Use index-space for fiber output coordinates, otherwise us world space for fiber output coordinates (from tensor file).', argstr='--index_space ')
    voxelize = traits.Either(traits.Bool, File(), hash_files=False, desc='Voxelize fiber into a label map (the labelmap filename is the argument of -V). The tensor file must be specified using -T for information about the size, origin, spacing of the image. The deformation is applied before the voxelization ', argstr='--voxelize %s')
    voxelize_count_fibers = traits.Bool(desc='Count number of fibers per-voxel instead of just setting to 1', argstr='--voxelize_count_fibers ')
    voxel_label = traits.Int(desc='Label for voxelized fiber', argstr='--voxel_label %d')
    verbose = traits.Bool(desc='produce verbose output', argstr='--verbose ')
    noDataChange = traits.Bool(desc='Do not change data ??? ', argstr='--noDataChange ')