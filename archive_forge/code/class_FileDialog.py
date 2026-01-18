import sys
from ..Qt import QtWidgets
class FileDialog(QtWidgets.QFileDialog):

    def __init__(self, *args):
        QtWidgets.QFileDialog.__init__(self, *args)
        if sys.platform == 'darwin':
            self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)