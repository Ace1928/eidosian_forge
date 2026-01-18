from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
class ConclusionPage(QtWidgets.QWizardPage):

    def __init__(self, parent=None):
        super(ConclusionPage, self).__init__(parent)
        self.setTitle('Conclusion')
        self.setPixmap(QtWidgets.QWizard.WatermarkPixmap, QtGui.QPixmap(':/images/watermark2.png'))
        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def initializePage(self):
        finishText = self.wizard().buttonText(QtWidgets.QWizard.FinishButton)
        finishText.replace('&', '')
        self.label.setText('Click %s to generate the class skeleton.' % finishText)