from nipype.interfaces.base import (
import os
class PETStandardUptakeValueComputationInputSpec(CommandLineInputSpec):
    petDICOMPath = Directory(desc='Input path to a directory containing a PET volume containing DICOM header information for SUV computation', exists=True, argstr='--petDICOMPath %s')
    petVolume = File(desc='Input PET volume for SUVbw computation (must be the same volume as pointed to by the DICOM path!).', exists=True, argstr='--petVolume %s')
    labelMap = File(desc='Input label volume containing the volumes of interest', exists=True, argstr='--labelMap %s')
    color = File(desc='Color table to to map labels to colors and names', exists=True, argstr='--color %s')
    csvFile = traits.Either(traits.Bool, File(), hash_files=False, desc='A file holding the output SUV values in comma separated lines, one per label. Optional.', argstr='--csvFile %s')
    OutputLabel = traits.Str(desc='List of labels for which SUV values were computed', argstr='--OutputLabel %s')
    OutputLabelValue = traits.Str(desc='List of label values for which SUV values were computed', argstr='--OutputLabelValue %s')
    SUVMax = traits.Str(desc='SUV max for each label', argstr='--SUVMax %s')
    SUVMean = traits.Str(desc='SUV mean for each label', argstr='--SUVMean %s')
    SUVMin = traits.Str(desc='SUV minimum for each label', argstr='--SUVMin %s')