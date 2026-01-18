from PySide2 import QtCore, QtGui, QtWidgets
def aboutToShowSaveAsMenu(self):
    currentText = self.textEdit.toPlainText()
    for action in self.saveAsActs:
        codecName = str(action.data())
        codec = QtCore.QTextCodec.codecForName(codecName)
        action.setVisible(codec and codec.canEncode(currentText))