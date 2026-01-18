import os
from ...base import (
class BRAINSABC(SEMLikeCommandLine):
    """title: Intra-subject registration, bias Correction, and tissue classification (BRAINS)

    category: Segmentation.Specialized

    description: Atlas-based tissue segmentation method.  This is an algorithmic extension of work done by XXXX at UNC and Utah XXXX need more description here.
    """
    input_spec = BRAINSABCInputSpec
    output_spec = BRAINSABCOutputSpec
    _cmd = ' BRAINSABC '
    _outputs_filenames = {'saveState': 'saveState.h5', 'outputLabels': 'outputLabels.nii.gz', 'atlasToSubjectTransform': 'atlasToSubjectTransform.h5', 'atlasToSubjectInitialTransform': 'atlasToSubjectInitialTransform.h5', 'outputDirtyLabels': 'outputDirtyLabels.nii.gz', 'outputVolumes': 'outputVolumes.nii.gz', 'outputDir': 'outputDir', 'implicitOutputs': 'implicitOutputs.nii.gz'}
    _redirect_x = False