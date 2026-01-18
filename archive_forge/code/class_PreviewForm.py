from PySide2 import QtCore, QtGui, QtWidgets
class PreviewForm(QtWidgets.QDialog):

    def __init__(self, parent):
        super(PreviewForm, self).__init__(parent)
        self.encodingComboBox = QtWidgets.QComboBox()
        encodingLabel = QtWidgets.QLabel('&Encoding:')
        encodingLabel.setBuddy(self.encodingComboBox)
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEdit.setReadOnly(True)
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.encodingComboBox.activated.connect(self.updateTextEdit)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(encodingLabel, 0, 0)
        mainLayout.addWidget(self.encodingComboBox, 0, 1)
        mainLayout.addWidget(self.textEdit, 1, 0, 1, 2)
        mainLayout.addWidget(buttonBox, 2, 0, 1, 2)
        self.setLayout(mainLayout)
        self.setWindowTitle('Choose Encoding')
        self.resize(400, 300)

    def setCodecList(self, codecs):
        self.encodingComboBox.clear()
        for codec in codecs:
            self.encodingComboBox.addItem(codec_name(codec), codec.mibEnum())

    def setEncodedData(self, data):
        self.encodedData = data
        self.updateTextEdit()

    def decodedString(self):
        return self.decodedStr

    def updateTextEdit(self):
        mib = self.encodingComboBox.itemData(self.encodingComboBox.currentIndex())
        codec = QtCore.QTextCodec.codecForMib(mib)
        data = QtCore.QTextStream(self.encodedData)
        data.setAutoDetectUnicode(False)
        data.setCodec(codec)
        self.decodedStr = data.readAll()
        self.textEdit.setPlainText(self.decodedStr)