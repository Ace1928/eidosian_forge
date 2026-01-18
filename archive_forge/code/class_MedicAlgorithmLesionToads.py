import os
from ..base import (
class MedicAlgorithmLesionToads(SEMLikeCommandLine):
    """Algorithm for simultaneous brain structures and MS lesion segmentation of MS Brains.

    The brain segmentation is topologically consistent and the algorithm can use multiple
    MR sequences as input data.

    References
    ----------
    N. Shiee, P.-L. Bazin, A.Z. Ozturk, P.A. Calabresi, D.S. Reich, D.L. Pham,
    "A Topology-Preserving Approach to the Segmentation of Brain Images with Multiple Sclerosis",
    NeuroImage, vol. 49, no. 2, pp. 1524-1535, 2010.

    """
    input_spec = MedicAlgorithmLesionToadsInputSpec
    output_spec = MedicAlgorithmLesionToadsOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.classification.MedicAlgorithmLesionToads '
    _outputs_filenames = {'outWM': 'outWM.nii', 'outHard': 'outHard.nii', 'outFilled': 'outFilled.nii', 'outMembership': 'outMembership.nii', 'outInhomogeneity': 'outInhomogeneity.nii', 'outCortical': 'outCortical.nii', 'outHard2': 'outHard2.nii', 'outLesion': 'outLesion.nii', 'outSulcal': 'outSulcal.nii'}
    _redirect_x = True