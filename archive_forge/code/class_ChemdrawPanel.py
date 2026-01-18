import sys
from qt import *
class ChemdrawPanel(QWidget):

    def __init__(self, parent=None, name='test', readOnly=0, size=(300, 300)):
        QWidget.__init__(self, parent, name)
        self.cdx = None
        self.resize(size[0], size[1])
        theClass = MakeActiveXClass(cdxModule.ChemDrawCtl, eventObj=self)
        self.cdx = theClass(self)
        if readOnly:
            self.cdx.ViewOnly = 1
        self.offset = 30
        self.label = QLabel(self, 'ChemDraw')
        self.label.setText(name)
        self.label.setAlignment(Qt.AlignHCenter)
        fnt = QApplication.font()
        fnt.setPointSize(14)
        self.label.setFont(fnt)

    def pullData(self, fmt='chemical/daylight-smiles'):
        data = self.cdx.GetData(fmt)
        return str(data)

    def setData(self, data, fmt='chemical/daylight-smiles'):
        self.cdx.Objects.Clear()
        res = self.cdx.SetData(fmt, data)
        return res

    def resizeEvent(self, evt):
        sz = evt.size()
        self.label.setGeometry(0, 0, sz.width(), self.offset)
        self.cdx.MoveWindow((0, self.offset, sz.width(), sz.height()), 1)

    def __del__(self):
        if self.cdx:
            self.cdx = None