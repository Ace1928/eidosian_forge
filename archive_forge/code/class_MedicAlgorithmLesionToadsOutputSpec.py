import os
from ..base import (
class MedicAlgorithmLesionToadsOutputSpec(TraitedSpec):
    outHard = File(desc='Hard segmentation', exists=True)
    outHard2 = File(desc='Hard segmentationfrom memberships', exists=True)
    outInhomogeneity = File(desc='Inhomogeneity Field', exists=True)
    outMembership = File(desc='Membership Functions', exists=True)
    outLesion = File(desc='Lesion Segmentation', exists=True)
    outSulcal = File(desc='Sulcal CSF Membership', exists=True)
    outCortical = File(desc='Cortical GM Membership', exists=True)
    outFilled = File(desc='Filled WM Membership', exists=True)
    outWM = File(desc='WM Mask', exists=True)