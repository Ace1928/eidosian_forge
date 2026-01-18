from PySide2.QtCore import (Qt, Signal, QRegExp, QModelIndex,
from PySide2.QtWidgets import (QWidget, QTabWidget, QMessageBox, QTableView,
from tablemodel import TableModel
from newaddresstab import NewAddressTab
from adddialogwidget import AddDialogWidget
class AddressWidget(QTabWidget):
    """ The central widget of the application. Most of the addressbook's
        functionality is contained in this class.
    """
    selectionChanged = Signal(QItemSelection)

    def __init__(self, parent=None):
        """ Initialize the AddressWidget. """
        super(AddressWidget, self).__init__(parent)
        self.tableModel = TableModel()
        self.newAddressTab = NewAddressTab()
        self.newAddressTab.sendDetails.connect(self.addEntry)
        self.addTab(self.newAddressTab, 'Address Book')
        self.setupTabs()

    def addEntry(self, name=None, address=None):
        """ Add an entry to the addressbook. """
        if name is None and address is None:
            addDialog = AddDialogWidget()
            if addDialog.exec_():
                name = addDialog.name
                address = addDialog.address
        address = {'name': name, 'address': address}
        addresses = self.tableModel.addresses[:]
        try:
            addresses.remove(address)
            QMessageBox.information(self, 'Duplicate Name', 'The name "%s" already exists.' % name)
        except ValueError:
            self.tableModel.insertRows(0)
            ix = self.tableModel.index(0, 0, QModelIndex())
            self.tableModel.setData(ix, address['name'], Qt.EditRole)
            ix = self.tableModel.index(0, 1, QModelIndex())
            self.tableModel.setData(ix, address['address'], Qt.EditRole)
            self.removeTab(self.indexOf(self.newAddressTab))
            tableView = self.currentWidget()
            tableView.resizeRowToContents(ix.row())

    def editEntry(self):
        """ Edit an entry in the addressbook. """
        tableView = self.currentWidget()
        proxyModel = tableView.model()
        selectionModel = tableView.selectionModel()
        indexes = selectionModel.selectedRows()
        for index in indexes:
            row = proxyModel.mapToSource(index).row()
            ix = self.tableModel.index(row, 0, QModelIndex())
            name = self.tableModel.data(ix, Qt.DisplayRole)
            ix = self.tableModel.index(row, 1, QModelIndex())
            address = self.tableModel.data(ix, Qt.DisplayRole)
        addDialog = AddDialogWidget()
        addDialog.setWindowTitle('Edit a Contact')
        addDialog.nameText.setReadOnly(True)
        addDialog.nameText.setText(name)
        addDialog.addressText.setText(address)
        if addDialog.exec_():
            newAddress = addDialog.address
            if newAddress != address:
                ix = self.tableModel.index(row, 1, QModelIndex())
                self.tableModel.setData(ix, newAddress, Qt.EditRole)

    def removeEntry(self):
        """ Remove an entry from the addressbook. """
        tableView = self.currentWidget()
        proxyModel = tableView.model()
        selectionModel = tableView.selectionModel()
        indexes = selectionModel.selectedRows()
        for index in indexes:
            row = proxyModel.mapToSource(index).row()
            self.tableModel.removeRows(row)
        if self.tableModel.rowCount() == 0:
            self.insertTab(0, self.newAddressTab, 'Address Book')

    def setupTabs(self):
        """ Setup the various tabs in the AddressWidget. """
        groups = ['ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQR', 'STU', 'VW', 'XYZ']
        for group in groups:
            proxyModel = QSortFilterProxyModel(self)
            proxyModel.setSourceModel(self.tableModel)
            proxyModel.setDynamicSortFilter(True)
            tableView = QTableView()
            tableView.setModel(proxyModel)
            tableView.setSortingEnabled(True)
            tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
            tableView.horizontalHeader().setStretchLastSection(True)
            tableView.verticalHeader().hide()
            tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tableView.setSelectionMode(QAbstractItemView.SingleSelection)
            reFilter = '^[%s].*' % group
            proxyModel.setFilterRegExp(QRegExp(reFilter, Qt.CaseInsensitive))
            proxyModel.setFilterKeyColumn(0)
            proxyModel.sort(0, Qt.AscendingOrder)
            viewselectionmodel = tableView.selectionModel()
            tableView.selectionModel().selectionChanged.connect(self.selectionChanged)
            self.addTab(tableView, group)

    def readFromFile(self, filename):
        """ Read contacts in from a file. """
        try:
            f = open(filename, 'rb')
            addresses = pickle.load(f)
        except IOError:
            QMessageBox.information(self, 'Unable to open file: %s' % filename)
        finally:
            f.close()
        if len(addresses) == 0:
            QMessageBox.information(self, 'No contacts in file: %s' % filename)
        else:
            for address in addresses:
                self.addEntry(address['name'], address['address'])

    def writeToFile(self, filename):
        """ Save all contacts in the model to a file. """
        try:
            f = open(filename, 'wb')
            pickle.dump(self.tableModel.addresses, f)
        except IOError:
            QMessageBox.information(self, 'Unable to open file: %s' % filename)
        finally:
            f.close()