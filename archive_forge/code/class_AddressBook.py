from PySide2 import QtCore, QtGui, QtWidgets
class AddressBook(QtWidgets.QWidget):
    NavigationMode, AddingMode, EditingMode = range(3)

    def __init__(self, parent=None):
        super(AddressBook, self).__init__(parent)
        self.contacts = SortedDict()
        self.oldName = ''
        self.oldAddress = ''
        self.currentMode = self.NavigationMode
        nameLabel = QtWidgets.QLabel('Name:')
        self.nameLine = QtWidgets.QLineEdit()
        self.nameLine.setReadOnly(True)
        addressLabel = QtWidgets.QLabel('Address:')
        self.addressText = QtWidgets.QTextEdit()
        self.addressText.setReadOnly(True)
        self.addButton = QtWidgets.QPushButton('&Add')
        self.editButton = QtWidgets.QPushButton('&Edit')
        self.editButton.setEnabled(False)
        self.removeButton = QtWidgets.QPushButton('&Remove')
        self.removeButton.setEnabled(False)
        self.submitButton = QtWidgets.QPushButton('&Submit')
        self.submitButton.hide()
        self.cancelButton = QtWidgets.QPushButton('&Cancel')
        self.cancelButton.hide()
        self.nextButton = QtWidgets.QPushButton('&Next')
        self.nextButton.setEnabled(False)
        self.previousButton = QtWidgets.QPushButton('&Previous')
        self.previousButton.setEnabled(False)
        self.addButton.clicked.connect(self.addContact)
        self.submitButton.clicked.connect(self.submitContact)
        self.editButton.clicked.connect(self.editContact)
        self.removeButton.clicked.connect(self.removeContact)
        self.cancelButton.clicked.connect(self.cancel)
        self.nextButton.clicked.connect(self.next)
        self.previousButton.clicked.connect(self.previous)
        buttonLayout1 = QtWidgets.QVBoxLayout()
        buttonLayout1.addWidget(self.addButton)
        buttonLayout1.addWidget(self.editButton)
        buttonLayout1.addWidget(self.removeButton)
        buttonLayout1.addWidget(self.submitButton)
        buttonLayout1.addWidget(self.cancelButton)
        buttonLayout1.addStretch()
        buttonLayout2 = QtWidgets.QHBoxLayout()
        buttonLayout2.addWidget(self.previousButton)
        buttonLayout2.addWidget(self.nextButton)
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(nameLabel, 0, 0)
        mainLayout.addWidget(self.nameLine, 0, 1)
        mainLayout.addWidget(addressLabel, 1, 0, QtCore.Qt.AlignTop)
        mainLayout.addWidget(self.addressText, 1, 1)
        mainLayout.addLayout(buttonLayout1, 1, 2)
        mainLayout.addLayout(buttonLayout2, 3, 1)
        self.setLayout(mainLayout)
        self.setWindowTitle('Simple Address Book')

    def addContact(self):
        self.oldName = self.nameLine.text()
        self.oldAddress = self.addressText.toPlainText()
        self.nameLine.clear()
        self.addressText.clear()
        self.updateInterface(self.AddingMode)

    def editContact(self):
        self.oldName = self.nameLine.text()
        self.oldAddress = self.addressText.toPlainText()
        self.updateInterface(self.EditingMode)

    def submitContact(self):
        name = self.nameLine.text()
        address = self.addressText.toPlainText()
        if name == '' or address == '':
            QtWidgets.QMessageBox.information(self, 'Empty Field', 'Please enter a name and address.')
            return
        if self.currentMode == self.AddingMode:
            if name not in self.contacts:
                self.contacts[name] = address
                QtWidgets.QMessageBox.information(self, 'Add Successful', '"%s" has been added to your address book.' % name)
            else:
                QtWidgets.QMessageBox.information(self, 'Add Unsuccessful', 'Sorry, "%s" is already in your address book.' % name)
                return
        elif self.currentMode == self.EditingMode:
            if self.oldName != name:
                if name not in self.contacts:
                    QtWidgets.QMessageBox.information(self, 'Edit Successful', '"%s" has been edited in your address book.' % self.oldName)
                    del self.contacts[self.oldName]
                    self.contacts[name] = address
                else:
                    QtWidgets.QMessageBox.information(self, 'Edit Unsuccessful', 'Sorry, "%s" is already in your address book.' % name)
                    return
            elif self.oldAddress != address:
                QtWidgets.QMessageBox.information(self, 'Edit Successful', '"%s" has been edited in your address book.' % name)
                self.contacts[name] = address
        self.updateInterface(self.NavigationMode)

    def cancel(self):
        self.nameLine.setText(self.oldName)
        self.addressText.setText(self.oldAddress)
        self.updateInterface(self.NavigationMode)

    def removeContact(self):
        name = self.nameLine.text()
        address = self.addressText.toPlainText()
        if name in self.contacts:
            button = QtWidgets.QMessageBox.question(self, 'Confirm Remove', 'Are you sure you want to remove "%s"?' % name, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if button == QtWidgets.QMessageBox.Yes:
                self.previous()
                del self.contacts[name]
                QtWidgets.QMessageBox.information(self, 'Remove Successful', '"%s" has been removed from your address book.' % name)
        self.updateInterface(self.NavigationMode)

    def next(self):
        name = self.nameLine.text()
        it = iter(self.contacts)
        try:
            while True:
                this_name, _ = it.next()
                if this_name == name:
                    next_name, next_address = it.next()
                    break
        except StopIteration:
            next_name, next_address = iter(self.contacts).next()
        self.nameLine.setText(next_name)
        self.addressText.setText(next_address)

    def previous(self):
        name = self.nameLine.text()
        prev_name = prev_address = None
        for this_name, this_address in self.contacts:
            if this_name == name:
                break
            prev_name = this_name
            prev_address = this_address
        else:
            self.nameLine.clear()
            self.addressText.clear()
            return
        if prev_name is None:
            for prev_name, prev_address in self.contacts:
                pass
        self.nameLine.setText(prev_name)
        self.addressText.setText(prev_address)

    def updateInterface(self, mode):
        self.currentMode = mode
        if self.currentMode in (self.AddingMode, self.EditingMode):
            self.nameLine.setReadOnly(False)
            self.nameLine.setFocus(QtCore.Qt.OtherFocusReason)
            self.addressText.setReadOnly(False)
            self.addButton.setEnabled(False)
            self.editButton.setEnabled(False)
            self.removeButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.previousButton.setEnabled(False)
            self.submitButton.show()
            self.cancelButton.show()
        elif self.currentMode == self.NavigationMode:
            if not self.contacts:
                self.nameLine.clear()
                self.addressText.clear()
            self.nameLine.setReadOnly(True)
            self.addressText.setReadOnly(True)
            self.addButton.setEnabled(True)
            number = len(self.contacts)
            self.editButton.setEnabled(number >= 1)
            self.removeButton.setEnabled(number >= 1)
            self.nextButton.setEnabled(number > 1)
            self.previousButton.setEnabled(number > 1)
            self.submitButton.hide()
            self.cancelButton.hide()