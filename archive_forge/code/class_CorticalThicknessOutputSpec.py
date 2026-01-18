import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class CorticalThicknessOutputSpec(TraitedSpec):
    BrainExtractionMask = File(exists=True, desc='brain extraction mask')
    ExtractedBrainN4 = File(exists=True, desc='extracted brain from N4 image')
    BrainSegmentation = File(exists=True, desc='brain segmentation image')
    BrainSegmentationN4 = File(exists=True, desc='N4 corrected image')
    BrainSegmentationPosteriors = OutputMultiPath(File(exists=True), desc='Posterior probability images')
    CorticalThickness = File(exists=True, desc='cortical thickness file')
    TemplateToSubject1GenericAffine = File(exists=True, desc='Template to subject affine')
    TemplateToSubject0Warp = File(exists=True, desc='Template to subject warp')
    SubjectToTemplate1Warp = File(exists=True, desc='Template to subject inverse warp')
    SubjectToTemplate0GenericAffine = File(exists=True, desc='Template to subject inverse affine')
    SubjectToTemplateLogJacobian = File(exists=True, desc='Template to subject log jacobian')
    CorticalThicknessNormedToTemplate = File(exists=True, desc='Normalized cortical thickness')
    BrainVolumes = File(exists=True, desc='Brain volumes as text')