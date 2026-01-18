import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
def _create_gradient_matrix(self, bvecs_file, bvals_file):
    _gradient_matrix_file = 'gradient_matrix.txt'
    bvals = [val for val in re.split('\\s+', open(bvals_file).readline().strip())]
    bvecs_f = open(bvecs_file)
    bvecs_x = [val for val in re.split('\\s+', bvecs_f.readline().strip())]
    bvecs_y = [val for val in re.split('\\s+', bvecs_f.readline().strip())]
    bvecs_z = [val for val in re.split('\\s+', bvecs_f.readline().strip())]
    bvecs_f.close()
    gradient_matrix_f = open(_gradient_matrix_file, 'w')
    for i in range(len(bvals)):
        if int(bvals[i]) == 0:
            continue
        gradient_matrix_f.write('%s %s %s\n' % (bvecs_x[i], bvecs_y[i], bvecs_z[i]))
    gradient_matrix_f.close()
    return _gradient_matrix_file