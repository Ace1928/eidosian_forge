from nipype.interfaces.base import (
import os
class BRAINSROIAuto(SEMLikeCommandLine):
    """title: Foreground masking (BRAINS)

    category: Segmentation.Specialized

    description: This tool uses a combination of otsu thresholding and a closing operations to identify the most prominent foreground region in an image.


    version: 2.4.1

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Hans J. Johnson, hans-johnson -at- uiowa.edu, http://wwww.psychiatry.uiowa.edu

    acknowledgements: Hans Johnson(1,3,4); Kent Williams(1); Gregory Harris(1), Vincent Magnotta(1,2,3);  Andriy Fedorov(5), fedorov -at- bwh.harvard.edu (Slicer integration); (1=University of Iowa Department of Psychiatry, 2=University of Iowa Department of Radiology, 3=University of Iowa Department of Biomedical Engineering, 4=University of Iowa Department of Electrical and Computer Engineering, 5=Surgical Planning Lab, Harvard)
    """
    input_spec = BRAINSROIAutoInputSpec
    output_spec = BRAINSROIAutoOutputSpec
    _cmd = 'BRAINSROIAuto '
    _outputs_filenames = {'outputROIMaskVolume': 'outputROIMaskVolume.nii', 'outputClippedVolumeROI': 'outputClippedVolumeROI.nii'}