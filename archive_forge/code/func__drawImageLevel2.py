import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def _drawImageLevel2(self, image, x1, y1, width=None, height=None):
    """At present we're handling only PIL"""
    if image.mode == 'L':
        imBitsPerComponent = 8
        imNumComponents = 1
        myimage = image
    elif image.mode == '1':
        myimage = image.convert('L')
        imNumComponents = 1
        myimage = image
    else:
        myimage = image.convert('RGB')
        imNumComponents = 3
        imBitsPerComponent = 8
    imwidth, imheight = myimage.size
    if not width:
        width = imwidth
    if not height:
        height = imheight
    self.code.extend(['gsave', '%s %s translate' % (x1, y1), '%s %s scale' % (width, height)])
    if imNumComponents == 3:
        self.code_append('/DeviceRGB setcolorspace')
    elif imNumComponents == 1:
        self.code_append('/DeviceGray setcolorspace')
    self.code_append('\n<<\n/ImageType 1\n/Width %d /Height %d  %% dimensions of source image\n/BitsPerComponent %d' % (imwidth, imheight, imBitsPerComponent))
    if imNumComponents == 1:
        self.code_append('/Decode [0 1]')
    if imNumComponents == 3:
        self.code_append('/Decode [0 1 0 1 0 1]  %% decode color values normally')
    self.code.extend(['/ImageMatrix [%s 0 0 %s 0 %s]' % (imwidth, -imheight, imheight), '/DataSource currentfile /ASCIIHexDecode filter', '>> % End image dictionary', 'image'])
    rawimage = (myimage.tobytes if hasattr(myimage, 'tobytes') else myimage.tostring)()
    hex_encoded = self._AsciiHexEncode(rawimage)
    outstream = StringIO(hex_encoded)
    dataline = outstream.read(78)
    while dataline != '':
        self.code_append(dataline)
        dataline = outstream.read(78)
    self.code_append('> % end of image data')
    self.code_append('grestore')