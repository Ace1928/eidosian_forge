import sys
from PySide2 import QtCore, QtGui, QtWidgets
class LocationDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(LocationDialog, self).__init__(parent)
        self.formatComboBox = QtWidgets.QComboBox()
        self.formatComboBox.addItem('Native')
        self.formatComboBox.addItem('INI')
        self.scopeComboBox = QtWidgets.QComboBox()
        self.scopeComboBox.addItem('User')
        self.scopeComboBox.addItem('System')
        self.organizationComboBox = QtWidgets.QComboBox()
        self.organizationComboBox.addItem('Trolltech')
        self.organizationComboBox.setEditable(True)
        self.applicationComboBox = QtWidgets.QComboBox()
        self.applicationComboBox.addItem('Any')
        self.applicationComboBox.addItem('Application Example')
        self.applicationComboBox.addItem('Assistant')
        self.applicationComboBox.addItem('Designer')
        self.applicationComboBox.addItem('Linguist')
        self.applicationComboBox.setEditable(True)
        self.applicationComboBox.setCurrentIndex(3)
        formatLabel = QtWidgets.QLabel('&Format:')
        formatLabel.setBuddy(self.formatComboBox)
        scopeLabel = QtWidgets.QLabel('&Scope:')
        scopeLabel.setBuddy(self.scopeComboBox)
        organizationLabel = QtWidgets.QLabel('&Organization:')
        organizationLabel.setBuddy(self.organizationComboBox)
        applicationLabel = QtWidgets.QLabel('&Application:')
        applicationLabel.setBuddy(self.applicationComboBox)
        self.locationsGroupBox = QtWidgets.QGroupBox('Setting Locations')
        self.locationsTable = QtWidgets.QTableWidget()
        self.locationsTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.locationsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.locationsTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.locationsTable.setColumnCount(2)
        self.locationsTable.setHorizontalHeaderLabels(('Location', 'Access'))
        self.locationsTable.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.locationsTable.horizontalHeader().resizeSection(1, 180)
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.formatComboBox.activated.connect(self.updateLocationsTable)
        self.scopeComboBox.activated.connect(self.updateLocationsTable)
        self.organizationComboBox.lineEdit().editingFinished.connect(self.updateLocationsTable)
        self.applicationComboBox.lineEdit().editingFinished.connect(self.updateLocationsTable)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        locationsLayout = QtWidgets.QVBoxLayout()
        locationsLayout.addWidget(self.locationsTable)
        self.locationsGroupBox.setLayout(locationsLayout)
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(formatLabel, 0, 0)
        mainLayout.addWidget(self.formatComboBox, 0, 1)
        mainLayout.addWidget(scopeLabel, 1, 0)
        mainLayout.addWidget(self.scopeComboBox, 1, 1)
        mainLayout.addWidget(organizationLabel, 2, 0)
        mainLayout.addWidget(self.organizationComboBox, 2, 1)
        mainLayout.addWidget(applicationLabel, 3, 0)
        mainLayout.addWidget(self.applicationComboBox, 3, 1)
        mainLayout.addWidget(self.locationsGroupBox, 4, 0, 1, 2)
        mainLayout.addWidget(self.buttonBox, 5, 0, 1, 2)
        self.setLayout(mainLayout)
        self.updateLocationsTable()
        self.setWindowTitle('Open Application Settings')
        self.resize(650, 400)

    def format(self):
        if self.formatComboBox.currentIndex() == 0:
            return QtCore.QSettings.NativeFormat
        else:
            return QtCore.QSettings.IniFormat

    def scope(self):
        if self.scopeComboBox.currentIndex() == 0:
            return QtCore.QSettings.UserScope
        else:
            return QtCore.QSettings.SystemScope

    def organization(self):
        return self.organizationComboBox.currentText()

    def application(self):
        if self.applicationComboBox.currentText() == 'Any':
            return ''
        return self.applicationComboBox.currentText()

    def updateLocationsTable(self):
        self.locationsTable.setUpdatesEnabled(False)
        self.locationsTable.setRowCount(0)
        for i in range(2):
            if i == 0:
                if self.scope() == QtCore.QSettings.SystemScope:
                    continue
                actualScope = QtCore.QSettings.UserScope
            else:
                actualScope = QtCore.QSettings.SystemScope
            for j in range(2):
                if j == 0:
                    if not self.application():
                        continue
                    actualApplication = self.application()
                else:
                    actualApplication = ''
                settings = QtCore.QSettings(self.format(), actualScope, self.organization(), actualApplication)
                row = self.locationsTable.rowCount()
                self.locationsTable.setRowCount(row + 1)
                item0 = QtWidgets.QTableWidgetItem()
                item0.setText(settings.fileName())
                item1 = QtWidgets.QTableWidgetItem()
                disable = not (settings.childKeys() or settings.childGroups())
                if row == 0:
                    if settings.isWritable():
                        item1.setText('Read-write')
                        disable = False
                    else:
                        item1.setText('Read-only')
                    self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(disable)
                else:
                    item1.setText('Read-only fallback')
                if disable:
                    item0.setFlags(item0.flags() & ~QtCore.Qt.ItemIsEnabled)
                    item1.setFlags(item1.flags() & ~QtCore.Qt.ItemIsEnabled)
                self.locationsTable.setItem(row, 0, item0)
                self.locationsTable.setItem(row, 1, item1)
        self.locationsTable.setUpdatesEnabled(True)