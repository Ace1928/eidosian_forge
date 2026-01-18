from PySide2.QtCore import (Qt, Signal)
from PySide2.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout)
from adddialogwidget import AddDialogWidget
class NewAddressTab(QWidget):
    """ An extra tab that prompts the user to add new contacts.
        To be displayed only when there are no contacts in the model.
    """
    sendDetails = Signal(str, str)

    def __init__(self, parent=None):
        super(NewAddressTab, self).__init__(parent)
        descriptionLabel = QLabel('There are no contacts in your address book.\nClick Add to add new contacts.')
        addButton = QPushButton('Add')
        layout = QVBoxLayout()
        layout.addWidget(descriptionLabel)
        layout.addWidget(addButton, 0, Qt.AlignCenter)
        self.setLayout(layout)
        addButton.clicked.connect(self.addEntry)

    def addEntry(self):
        addDialog = AddDialogWidget()
        if addDialog.exec_():
            name = addDialog.name
            address = addDialog.address
            self.sendDetails.emit(name, address)