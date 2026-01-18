import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class ODFReconInputSpec(CommandLineInputSpec):
    DWI = File(desc='Input raw data', argstr='%s', exists=True, mandatory=True, position=1)
    n_directions = traits.Int(desc='Number of directions', argstr='%s', mandatory=True, position=2)
    n_output_directions = traits.Int(desc='Number of output directions', argstr='%s', mandatory=True, position=3)
    out_prefix = traits.Str('odf', desc='Output file prefix', argstr='%s', usedefault=True, position=4)
    matrix = File(argstr='-mat %s', exists=True, desc='use given file as reconstruction matrix.', mandatory=True)
    n_b0 = traits.Int(argstr='-b0 %s', desc='number of b0 scans. by default the program gets this information\nfrom the number of directions and number of volumes in\nthe raw data. useful when dealing with incomplete raw\ndata set or only using part of raw data set to reconstruct', mandatory=True)
    output_type = traits.Enum('nii', 'analyze', 'ni1', 'nii.gz', argstr='-ot %s', desc='output file type', usedefault=True)
    sharpness = traits.Float(desc='smooth or sharpen the raw data. factor > 0 is smoothing.\nfactor < 0 is sharpening. default value is 0\nNOTE: this option applies to DSI study only', argstr='-s %f')
    filter = traits.Bool(desc='apply a filter (e.g. high pass) to the raw image', argstr='-f')
    subtract_background = traits.Bool(desc='subtract the background value before reconstruction', argstr='-bg')
    dsi = traits.Bool(desc='indicates that the data is dsi', argstr='-dsi')
    output_entropy = traits.Bool(desc='output entropy map', argstr='-oe')
    image_orientation_vectors = traits.List(traits.Float(), minlen=6, maxlen=6, desc='specify image orientation vectors. if just one argument given,\nwill treat it as filename and read the orientation vectors from\nthe file. if 6 arguments are given, will treat them as 6 float\nnumbers and construct the 1st and 2nd vector and calculate the 3rd\none automatically.\nthis information will be used to determine image orientation,\nas well as to adjust gradient vectors with oblique angle when', argstr='-iop %f')
    oblique_correction = traits.Bool(desc='when oblique angle(s) applied, some SIEMENS dti protocols do not\nadjust gradient accordingly, thus it requires adjustment for correct\ndiffusion tensor calculation', argstr='-oc')