from nipype.interfaces.base import (
import os
class GrayscaleModelMakerInputSpec(CommandLineInputSpec):
    InputVolume = File(position=-2, desc='Volume containing the input grayscale data.', exists=True, argstr='%s')
    OutputGeometry = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output that contains geometry model.', argstr='%s')
    threshold = traits.Float(desc='Grayscale threshold of isosurface. The resulting surface of triangles separates the volume into voxels that lie above (inside) and below (outside) the threshold.', argstr='--threshold %f')
    name = traits.Str(desc='Name to use for this model.', argstr='--name %s')
    smooth = traits.Int(desc='Number of smoothing iterations. If 0, no smoothing will be done.', argstr='--smooth %d')
    decimate = traits.Float(desc='Target reduction during decimation, as a decimal percentage reduction in the number of polygons. If 0, no decimation will be done.', argstr='--decimate %f')
    splitnormals = traits.Bool(desc='Splitting normals is useful for visualizing sharp features. However it creates holes in surfaces which affect measurements', argstr='--splitnormals ')
    pointnormals = traits.Bool(desc='Calculate the point normals? Calculated point normals make the surface appear smooth. Without point normals, the surface will appear faceted.', argstr='--pointnormals ')