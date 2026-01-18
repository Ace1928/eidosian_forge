from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
class CodeStylePage(QtWidgets.QWizardPage):

    def __init__(self, parent=None):
        super(CodeStylePage, self).__init__(parent)
        self.setTitle('Code Style Options')
        self.setSubTitle('Choose the formatting of the generated code.')
        self.setPixmap(QtWidgets.QWizard.LogoPixmap, QtGui.QPixmap(':/images/logo2.png'))
        commentCheckBox = QtWidgets.QCheckBox('&Start generated files with a comment')
        commentCheckBox.setChecked(True)
        protectCheckBox = QtWidgets.QCheckBox('&Protect header file against multiple inclusions')
        protectCheckBox.setChecked(True)
        macroNameLabel = QtWidgets.QLabel('&Macro name:')
        self.macroNameLineEdit = QtWidgets.QLineEdit()
        macroNameLabel.setBuddy(self.macroNameLineEdit)
        self.includeBaseCheckBox = QtWidgets.QCheckBox('&Include base class definition')
        self.baseIncludeLabel = QtWidgets.QLabel('Base class include:')
        self.baseIncludeLineEdit = QtWidgets.QLineEdit()
        self.baseIncludeLabel.setBuddy(self.baseIncludeLineEdit)
        protectCheckBox.toggled.connect(macroNameLabel.setEnabled)
        protectCheckBox.toggled.connect(self.macroNameLineEdit.setEnabled)
        self.includeBaseCheckBox.toggled.connect(self.baseIncludeLabel.setEnabled)
        self.includeBaseCheckBox.toggled.connect(self.baseIncludeLineEdit.setEnabled)
        self.registerField('comment', commentCheckBox)
        self.registerField('protect', protectCheckBox)
        self.registerField('macroName', self.macroNameLineEdit)
        self.registerField('includeBase', self.includeBaseCheckBox)
        self.registerField('baseInclude', self.baseIncludeLineEdit)
        layout = QtWidgets.QGridLayout()
        layout.setColumnMinimumWidth(0, 20)
        layout.addWidget(commentCheckBox, 0, 0, 1, 3)
        layout.addWidget(protectCheckBox, 1, 0, 1, 3)
        layout.addWidget(macroNameLabel, 2, 1)
        layout.addWidget(self.macroNameLineEdit, 2, 2)
        layout.addWidget(self.includeBaseCheckBox, 3, 0, 1, 3)
        layout.addWidget(self.baseIncludeLabel, 4, 1)
        layout.addWidget(self.baseIncludeLineEdit, 4, 2)
        self.setLayout(layout)

    def initializePage(self):
        className = self.field('className')
        self.macroNameLineEdit.setText(className.upper() + '_H')
        baseClass = self.field('baseClass')
        is_baseClass = bool(baseClass)
        self.includeBaseCheckBox.setChecked(is_baseClass)
        self.includeBaseCheckBox.setEnabled(is_baseClass)
        self.baseIncludeLabel.setEnabled(is_baseClass)
        self.baseIncludeLineEdit.setEnabled(is_baseClass)
        if not is_baseClass:
            self.baseIncludeLineEdit.clear()
        elif QtCore.QRegExp('Q[A-Z].*').exactMatch(baseClass):
            self.baseIncludeLineEdit.setText('<' + baseClass + '>')
        else:
            self.baseIncludeLineEdit.setText('"' + baseClass.lower() + '.h"')