import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class UndumpInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dUndump, whose geometry will determinethe geometry of the output', argstr='-master %s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(desc='output image file name', argstr='-prefix %s', name_source='in_file')
    mask_file = File(desc='mask image file name. Only voxels that are nonzero in the mask can be set.', argstr='-mask %s')
    datatype = traits.Enum('short', 'float', 'byte', desc='set output file datatype', argstr='-datum %s')
    default_value = traits.Float(desc='default value stored in each input voxel that does not have a value supplied in the input file', argstr='-dval %f')
    fill_value = traits.Float(desc='value, used for each voxel in the output dataset that is NOT listed in the input file', argstr='-fval %f')
    coordinates_specification = traits.Enum('ijk', 'xyz', desc='Coordinates in the input file as index triples (i, j, k) or spatial coordinates (x, y, z) in mm', argstr='-%s')
    srad = traits.Float(desc='radius in mm of the sphere that will be filled about each input (x,y,z) or (i,j,k) voxel. If the radius is not given, or is 0, then each input data line sets the value in only one voxel.', argstr='-srad %f')
    orient = traits.Tuple(traits.Enum('R', 'L'), traits.Enum('A', 'P'), traits.Enum('I', 'S'), desc="Specifies the coordinate order used by -xyz. The code must be 3 letters, one each from the pairs {R,L} {A,P} {I,S}.  The first letter gives the orientation of the x-axis, the second the orientation of the y-axis, the third the z-axis: R = right-to-left         L = left-to-right A = anterior-to-posterior P = posterior-to-anterior I = inferior-to-superior  S = superior-to-inferior If -orient isn't used, then the coordinate order of the -master (in_file) dataset is used to interpret (x,y,z) inputs.", argstr='-orient %s')
    head_only = traits.Bool(desc='create only the .HEAD file which gets exploited by the AFNI matlab library function New_HEAD.m', argstr='-head_only')