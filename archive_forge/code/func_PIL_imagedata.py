import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
def PIL_imagedata(self):
    import zlib
    image = self.image
    if image.format == 'JPEG':
        fp = image.fp
        fp.seek(0)
        return self._jpg_imagedata(fp)
    self.source = 'PIL'
    bpc = 8
    if image.mode == 'CMYK':
        myimage = image
        colorSpace = 'DeviceCMYK'
        bpp = 4
    elif image.mode == '1':
        myimage = image
        colorSpace = 'DeviceGray'
        bpp = 1
        bpc = 1
    elif image.mode == 'L':
        myimage = image
        colorSpace = 'DeviceGray'
        bpp = 1
    else:
        myimage = image.convert('RGB')
        colorSpace = 'RGB'
        bpp = 3
    imgwidth, imgheight = myimage.size
    imagedata = ['BI /W %d /H %d /BPC %d /CS /%s /F [%s/Fl] ID' % (imgwidth, imgheight, bpc, colorSpace, rl_config.useA85 and '/A85 ' or '')]
    raw = (myimage.tobytes if hasattr(myimage, 'tobytes') else myimage.tostring)()
    rowstride = imgwidth * bpc * bpp + 7 >> 3
    assert len(raw) == rowstride * imgheight, 'Wrong amount of data for image'
    data = zlib.compress(raw)
    if rl_config.useA85:
        data = asciiBase85Encode(data)
    pdfutils._chunker(data, imagedata)
    imagedata.append('EI')
    return (imagedata, imgwidth, imgheight)