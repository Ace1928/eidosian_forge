from nipype.interfaces.base import (
import os
class ProbeVolumeWithModel(SEMLikeCommandLine):
    """title: Probe Volume With Model

    category: Surface Models

    description: Paint a model by a volume (using vtkProbeFilter).

    version: 0.1.0.$Revision: 1892 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ProbeVolumeWithModel

    contributor: Lauren O'Donnell (SPL, BWH)

    acknowledgements: BWH, NCIGT/LMI
    """
    input_spec = ProbeVolumeWithModelInputSpec
    output_spec = ProbeVolumeWithModelOutputSpec
    _cmd = 'ProbeVolumeWithModel '
    _outputs_filenames = {'OutputModel': 'OutputModel.vtk'}