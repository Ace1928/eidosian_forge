from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
class IntroPage(QtWidgets.QWizardPage):

    def __init__(self, parent=None):
        super(IntroPage, self).__init__(parent)
        self.setTitle('Introduction')
        self.setPixmap(QtWidgets.QWizard.WatermarkPixmap, QtGui.QPixmap(':/images/watermark1.png'))
        label = QtWidgets.QLabel('This wizard will generate a skeleton C++ class definition, including a few functions. You simply need to specify the class name and set a few options to produce a header file and an implementation file for your new C++ class.')
        label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)