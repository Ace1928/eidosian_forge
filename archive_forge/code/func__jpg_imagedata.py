import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
def _jpg_imagedata(self, imageFile):
    info = pdfutils.readJPEGInfo(imageFile)
    self.source = 'JPEG'
    imgwidth, imgheight = (info[0], info[1])
    if info[2] == 1:
        colorSpace = 'DeviceGray'
    elif info[2] == 3:
        colorSpace = 'DeviceRGB'
    else:
        colorSpace = 'DeviceCMYK'
    imageFile.seek(0)
    imagedata = []
    imagedata.append('BI /W %d /H %d /BPC 8 /CS /%s /F [%s/DCT] ID' % (imgwidth, imgheight, colorSpace, rl_config.useA85 and '/A85 ' or ''))
    data = imageFile.read()
    if rl_config.useA85:
        data = asciiBase85Encode(data)
    pdfutils._chunker(data, imagedata)
    imagedata.append('EI')
    return (imagedata, imgwidth, imgheight)