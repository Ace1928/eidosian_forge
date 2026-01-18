import os
from ....base import (
class fibertrack(SEMLikeCommandLine):
    """title: FiberTrack (DTIProcess)

    category: Diffusion.Tractography

    description: This program implements a simple streamline tractography method based on the principal eigenvector of the tensor field. A fourth order Runge-Kutta integration rule used to advance the streamlines.
    As a first parameter you have to input the tensor field (with the --input_tensor_file option). Then the region of interest image file is set with the --input_roi_file. Next you want to set the output fiber file name after the --output_fiber_file option.
    You can specify the label value in the input_roi_file with the --target_label, --source_label and  --fobidden_label options. By default target label is 1, source label is 2 and forbidden label is 0. The source label is where the streamlines are seeded, the target label defines the voxels through which the fibers must pass by to be kept in the final fiber file and the forbidden label defines the voxels where the streamlines are stopped if they pass through it. There is also a --whole_brain option which, if enabled, consider both target and source labels of the roi image as target labels and all the voxels of the image are considered as sources.
    During the tractography, the --fa_min parameter is used as the minimum value needed at different voxel for the tracking to keep going along a streamline. The --step_size parameter is used for each iteration of the tracking algorithm and defines the length of each step. The --max_angle option defines the maximum angle allowed between two successive segments along the tracked fiber.

    version: 1.1.0

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/DTIProcess

    license: Copyright (c)  Casey Goodlett. All rights reserved.
      See http://www.ia.unc.edu/dev/Copyright.htm for details.
         This software is distributed WITHOUT ANY WARRANTY; without even
         the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
         PURPOSE.  See the above copyright notices for more information.

    contributor: Casey Goodlett

    acknowledgements: Hans Johnson(1,3,4); Kent Williams(1); (1=University of Iowa Department of Psychiatry, 3=University of Iowa Department of Biomedical Engineering, 4=University of Iowa Department of Electrical and Computer Engineering) provided conversions to make DTIProcess compatible with Slicer execution, and simplified the stand-alone build requirements by removing the dependencies on boost and a fortran compiler.
    """
    input_spec = fibertrackInputSpec
    output_spec = fibertrackOutputSpec
    _cmd = ' fibertrack '
    _outputs_filenames = {'output_fiber_file': 'output_fiber_file.vtk'}
    _redirect_x = False