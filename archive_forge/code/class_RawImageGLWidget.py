import numpy
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption, getCupy
from ..Qt import QtCore, QtGui, QtWidgets
class RawImageGLWidget(QOpenGLWidget):
    """
        Similar to RawImageWidget, but uses a GL widget to do all drawing.
        Performance varies between platforms; see examples/VideoSpeedTest for benchmarking.

        Checks if setConfigOptions(imageAxisOrder='row-major') was set.
        """

    def __init__(self, parent=None, scaled=False):
        QOpenGLWidget.__init__(self, parent)
        self.scaled = scaled
        self.image = None
        self.uploaded = False
        self.smooth = False
        self.opts = None

    def setImage(self, img, *args, **kargs):
        """
            img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
            Extra arguments are sent to functions.makeARGB
            """
        if getConfigOption('imageAxisOrder') == 'col-major':
            img = img.swapaxes(0, 1)
        self.opts = (img, args, kargs)
        self.image = None
        self.uploaded = False
        self.update()

    def initializeGL(self):
        self.texture = glGenTextures(1)

    def uploadTexture(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        if self.smooth:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        h, w = self.image.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image)
        glDisable(GL_TEXTURE_2D)
        self.uploaded = True

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.image is None:
            if self.opts is None:
                return
            img, args, kwds = self.opts
            kwds['useRGBA'] = True
            self.image, _ = fn.makeARGB(img, *args, **kwds)
        if not self.uploaded:
            self.uploadTexture()
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex3f(-1, -1, 0)
        glTexCoord2f(1, 1)
        glVertex3f(1, -1, 0)
        glTexCoord2f(1, 0)
        glVertex3f(1, 1, 0)
        glTexCoord2f(0, 0)
        glVertex3f(-1, 1, 0)
        glEnd()
        glDisable(GL_TEXTURE_2D)