import os
import sys
import numpy as np
from numpy.testing import assert_equal, assert_array_less
import skvideo
import skvideo.datasets
import skvideo.io
def _Gray2RGBHack_Helper(pix_fmt):
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0
    outputfile = sys._getframe().f_code.co_name + '.yuv'
    bits = 16 if pix_fmt[-4:][0:2] == '16' else 8
    outputdata = np.random.random(size=(1, 8, 8, 1)) * ((1 << bits) - 1)
    if pix_fmt[0:2] == 'ya':
        outputdata = np.concatenate((outputdata, np.zeros_like(outputdata)), axis=3)
    if bits == 16:
        outputdata = outputdata.astype(np.uint16)
    else:
        outputdata = outputdata.astype(np.uint8)
    T, N, M, C = outputdata.shape
    writer = skvideo.io.LibAVWriter(outputfile, verbosity=0)
    for i in range(T):
        writer.writeFrame(outputdata[i])
    writer.close()
    reader = skvideo.io.LibAVReader(outputfile, inputdict={'-s': '{}x{}'.format(M, N)}, verbosity=0)
    assert_equal(reader.getShape()[0:3], outputdata.shape[0:3])
    inputdata = np.empty(reader.getShape(), dtype=np.uint16)
    for i, frame in enumerate(reader.nextFrame()):
        inputdata[i] = frame * (1 << bits - 8)
    reader.close()
    assert_array_less(np.abs(inputdata[:, :, :, 0].astype('int32') - outputdata[:, :, :, 0].astype('int32')), 1 << bits - 7)
    os.remove(outputfile)