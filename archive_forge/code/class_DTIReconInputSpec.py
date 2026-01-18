import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTIReconInputSpec(CommandLineInputSpec):
    DWI = File(desc='Input diffusion volume', argstr='%s', exists=True, mandatory=True, position=1)
    out_prefix = traits.Str('dti', desc='Output file prefix', argstr='%s', usedefault=True, position=2)
    output_type = traits.Enum('nii', 'analyze', 'ni1', 'nii.gz', argstr='-ot %s', desc='output file type', usedefault=True)
    bvecs = File(exists=True, desc='b vectors file', argstr='-gm %s', mandatory=True)
    bvals = File(exists=True, desc='b values file', mandatory=True)
    n_averages = traits.Int(desc='Number of averages', argstr='-nex %s')
    image_orientation_vectors = traits.List(traits.Float(), minlen=6, maxlen=6, desc='Specify image orientation vectors. if just one argument given,\nwill treat it as filename and read the orientation vectors from\nthe file. If 6 arguments are given, will treat them as 6 float\nnumbers and construct the 1st and 2nd vector and calculate the 3rd\none automatically.\nThis information will be used to determine image orientation,\nas well as to adjust gradient vectors with oblique angle when.', argstr='-iop %f')
    oblique_correction = traits.Bool(desc='When oblique angle(s) applied, some SIEMENS DTI protocols do not\nadjust gradient accordingly, thus it requires adjustment for correct\ndiffusion tensor calculation', argstr='-oc')
    b0_threshold = traits.Float(desc='Program will use b0 image with the given threshold to mask out high\nbackground of fa/adc maps. by default it will calculate threshold\nautomatically. but if it failed, you need to set it manually.', argstr='-b0_th')