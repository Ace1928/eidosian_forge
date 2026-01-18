from nipype.interfaces.base import (
import os
class BRAINSFit(SEMLikeCommandLine):
    """title: General Registration (BRAINS)

    category: Registration

    description: Register a three-dimensional volume to a reference volume (Mattes Mutual Information by default). Described in BRAINSFit: Mutual Information Registrations of Whole-Brain 3D Images, Using the Insight Toolkit, Johnson H.J., Harris G., Williams K., The Insight Journal, 2007. http://hdl.handle.net/1926/1291

    version: 3.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:BRAINSFit

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Hans J. Johnson, hans-johnson -at- uiowa.edu, http://wwww.psychiatry.uiowa.edu

    acknowledgements: Hans Johnson(1,3,4); Kent Williams(1); Gregory Harris(1), Vincent Magnotta(1,2,3);  Andriy Fedorov(5) 1=University of Iowa Department of Psychiatry, 2=University of Iowa Department of Radiology, 3=University of Iowa Department of Biomedical Engineering, 4=University of Iowa Department of Electrical and Computer Engineering, 5=Surgical Planning Lab, Harvard
    """
    input_spec = BRAINSFitInputSpec
    output_spec = BRAINSFitOutputSpec
    _cmd = 'BRAINSFit '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii', 'bsplineTransform': 'bsplineTransform.mat', 'outputTransform': 'outputTransform.mat', 'outputFixedVolumeROI': 'outputFixedVolumeROI.nii', 'strippedOutputTransform': 'strippedOutputTransform.mat', 'outputMovingVolumeROI': 'outputMovingVolumeROI.nii', 'linearTransform': 'linearTransform.mat'}