import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class ODFTrackerInputSpec(CommandLineInputSpec):
    max = File(exists=True, mandatory=True)
    ODF = File(exists=True, mandatory=True)
    input_data_prefix = traits.Str('odf', desc='recon data prefix', argstr='%s', usedefault=True, position=0)
    out_file = File('tracks.trk', desc='output track file', argstr='%s', usedefault=True, position=1)
    input_output_type = traits.Enum('nii', 'analyze', 'ni1', 'nii.gz', argstr='-it %s', desc='input and output file type', usedefault=True)
    runge_kutta2 = traits.Bool(argstr='-rk2', desc='use 2nd order Runge-Kutta method for tracking.\ndefault tracking method is non-interpolate streamline')
    step_length = traits.Float(argstr='-l %f', desc='set step length, in the unit of minimum voxel size.\ndefault value is 0.1.')
    angle_threshold = traits.Float(argstr='-at %f', desc='set angle threshold. default value is 35 degree for\ndefault tracking method and 25 for rk2')
    random_seed = traits.Int(argstr='-rseed %s', desc='use random location in a voxel instead of the center of the voxel\nto seed. can also define number of seed per voxel. default is 1')
    invert_x = traits.Bool(argstr='-ix', desc='invert x component of the vector')
    invert_y = traits.Bool(argstr='-iy', desc='invert y component of the vector')
    invert_z = traits.Bool(argstr='-iz', desc='invert z component of the vector')
    swap_xy = traits.Bool(argstr='-sxy', desc='swap x and y vectors while tracking')
    swap_yz = traits.Bool(argstr='-syz', desc='swap y and z vectors while tracking')
    swap_zx = traits.Bool(argstr='-szx', desc='swap x and z vectors while tracking')
    disc = traits.Bool(argstr='-disc', desc='use disc tracking')
    mask1_file = File(desc='first mask image', mandatory=True, argstr='-m %s', position=2)
    mask1_threshold = traits.Float(desc='threshold value for the first mask image, if not given, the program will try automatically find the threshold', position=3)
    mask2_file = File(desc='second mask image', argstr='-m2 %s', position=4)
    mask2_threshold = traits.Float(desc='threshold value for the second mask image, if not given, the program will try automatically find the threshold', position=5)
    limit = traits.Int(argstr='-limit %d', desc='in some special case, such as heart data, some track may go into\ninfinite circle and take long time to stop. this option allows\nsetting a limit for the longest tracking steps (voxels)')
    dsi = traits.Bool(argstr='-dsi', desc='specify the input odf data is dsi. because dsi recon uses fixed\npre-calculated matrix, some special orientation patch needs to\nbe applied to keep dti/dsi/q-ball consistent.')
    image_orientation_vectors = traits.List(traits.Float(), minlen=6, maxlen=6, desc='specify image orientation vectors. if just one argument given,\nwill treat it as filename and read the orientation vectors from\nthe file. if 6 arguments are given, will treat them as 6 float\nnumbers and construct the 1st and 2nd vector and calculate the 3rd\none automatically.\nthis information will be used to determine image orientation,\nas well as to adjust gradient vectors with oblique angle when', argstr='-iop %f')
    slice_order = traits.Int(argstr='-sorder %d', desc='set the slice order. 1 means normal, -1 means reversed. default value is 1')
    voxel_order = traits.Enum('RAS', 'RPS', 'RAI', 'RPI', 'LAI', 'LAS', 'LPS', 'LPI', argstr='-vorder %s', desc='specify the voxel order in RL/AP/IS (human brain) reference. must be\n3 letters with no space in between.\nfor example, RAS means the voxel row is from L->R, the column\nis from P->A and the slice order is from I->S.\nby default voxel order is determined by the image orientation\n(but NOT guaranteed to be correct because of various standards).\nfor example, siemens axial image is LPS, coronal image is LIP and\nsagittal image is PIL.\nthis information also is NOT needed for tracking but will be saved\nin the track file and is essential for track display to map onto\nthe right coordinates')