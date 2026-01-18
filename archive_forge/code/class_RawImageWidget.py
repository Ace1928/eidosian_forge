import numpy
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption, getCupy
from ..Qt import QtCore, QtGui, QtWidgets
class RawImageWidget(QtWidgets.QWidget):
    """
    Widget optimized for very fast video display.
    Generally using an ImageItem inside GraphicsView is fast enough.
    On some systems this may provide faster video. See the VideoSpeedTest example for benchmarking.
    """

    def __init__(self, parent=None, scaled=False):
        """
        Setting scaled=True will cause the entire image to be displayed within the boundaries of the widget.
        This also greatly reduces the speed at which it will draw frames.
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding))
        self.scaled = scaled
        self.opts = None
        self.image = None
        self._cp = getCupy()

    def setImage(self, img, *args, **kargs):
        """
        img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
        Extra arguments are sent to functions.makeARGB
        """
        if getConfigOption('imageAxisOrder') == 'col-major':
            img = img.swapaxes(0, 1)
        self.opts = (img, args, kargs)
        self.image = None
        self.update()

    def paintEvent(self, ev):
        if self.opts is None:
            return
        if self.image is None:
            img = self.opts[0]
            xp = self._cp.get_array_module(img) if self._cp else numpy
            qimage = None
            if not self.opts[1] and {'levels', 'lut'}.issuperset(self.opts[2]) and (not (img.dtype.kind == 'f' and xp.isnan(img.min()))):
                qimage = functions_qimage.try_make_qimage(img, levels=self.opts[2].get('levels'), lut=self.opts[2].get('lut'))
            if qimage is None:
                argb, alpha = fn.makeARGB(self.opts[0], *self.opts[1], **self.opts[2])
                if self._cp and self._cp.get_array_module(argb) == self._cp:
                    argb = argb.get()
                qimage = fn.ndarray_to_qimage(argb, QtGui.QImage.Format.Format_ARGB32)
            self.image = qimage
            self.opts = ()
        p = QtGui.QPainter(self)
        if self.scaled:
            rect = self.rect()
            ar = rect.width() / float(rect.height())
            imar = self.image.width() / float(self.image.height())
            if ar > imar:
                rect.setWidth(int(rect.width() * imar / ar))
            else:
                rect.setHeight(int(rect.height() * ar / imar))
            p.drawImage(rect, self.image)
        else:
            p.drawImage(QtCore.QPointF(), self.image)
        p.end()