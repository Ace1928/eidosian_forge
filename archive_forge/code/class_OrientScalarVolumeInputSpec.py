from nipype.interfaces.base import (
import os
class OrientScalarVolumeInputSpec(CommandLineInputSpec):
    inputVolume1 = File(position=-2, desc='Input volume 1', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='The oriented volume', argstr='%s')
    orientation = traits.Enum('Axial', 'Coronal', 'Sagittal', 'RIP', 'LIP', 'RSP', 'LSP', 'RIA', 'LIA', 'RSA', 'LSA', 'IRP', 'ILP', 'SRP', 'SLP', 'IRA', 'ILA', 'SRA', 'SLA', 'RPI', 'LPI', 'RAI', 'LAI', 'RPS', 'LPS', 'RAS', 'LAS', 'PRI', 'PLI', 'ARI', 'ALI', 'PRS', 'PLS', 'ARS', 'ALS', 'IPR', 'SPR', 'IAR', 'SAR', 'IPL', 'SPL', 'IAL', 'SAL', 'PIR', 'PSR', 'AIR', 'ASR', 'PIL', 'PSL', 'AIL', 'ASL', desc='Orientation choices', argstr='--orientation %s')