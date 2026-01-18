from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
class ProgressDialog(QtWidgets.QProgressDialog):
    """
    Extends QProgressDialog:
    
      * Adds context management so the dialog may be used in `with` statements
      * Allows nesting multiple progress dialogs

    Example::

        with ProgressDialog("Processing..", minVal, maxVal) as dlg:
            # do stuff
            dlg.setValue(i)   ## could also use dlg += 1
            if dlg.wasCanceled():
                raise Exception("Processing canceled by user")
    """
    allDialogs = []

    def __init__(self, labelText, minimum=0, maximum=100, cancelText='Cancel', parent=None, wait=250, busyCursor=False, disable=False, nested=False):
        """
        ============== ================================================================
        **Arguments:**
        labelText      (required)
        cancelText     Text to display on cancel button, or None to disable it.
        minimum
        maximum
        parent       
        wait           Length of time (im ms) to wait before displaying dialog
        busyCursor     If True, show busy cursor until dialog finishes
        disable        If True, the progress dialog will not be displayed
                       and calls to wasCanceled() will always return False.
                       If ProgressDialog is entered from a non-gui thread, it will
                       always be disabled.
        nested         (bool) If True, then this progress bar will be displayed inside
                       any pre-existing progress dialogs that also allow nesting.
        ============== ================================================================
        """
        self.nestedLayout = None
        self._nestableWidgets = None
        self._nestingReady = False
        self._topDialog = None
        self._subBars = []
        self.nested = nested
        self._lastProcessEvents = None
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        self.disabled = disable or not isGuiThread
        if self.disabled:
            return
        noCancel = False
        if cancelText is None:
            cancelText = ''
            noCancel = True
        self.busyCursor = busyCursor
        QtWidgets.QProgressDialog.__init__(self, labelText, cancelText, minimum, maximum, parent)
        if nested is True and len(ProgressDialog.allDialogs) > 0:
            self.setMinimumDuration(2 ** 30)
        else:
            self.setMinimumDuration(wait)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setValue(self.minimum())
        if noCancel:
            self.setCancelButton(None)

    def __enter__(self):
        if self.disabled:
            return self
        if self.busyCursor:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        if self.nested and len(ProgressDialog.allDialogs) > 0:
            topDialog = ProgressDialog.allDialogs[0]
            topDialog._addSubDialog(self)
            self._topDialog = topDialog
            topDialog.canceled.connect(self.cancel)
        ProgressDialog.allDialogs.append(self)
        return self

    def __exit__(self, exType, exValue, exTrace):
        if self.disabled:
            return
        if self.busyCursor:
            QtWidgets.QApplication.restoreOverrideCursor()
        if self._topDialog is not None:
            self._topDialog._removeSubDialog(self)
        ProgressDialog.allDialogs.pop(-1)
        self.setValue(self.maximum())

    def __iadd__(self, val):
        """Use inplace-addition operator for easy incrementing."""
        if self.disabled:
            return self
        self.setValue(self.value() + val)
        return self

    def _addSubDialog(self, dlg):
        self._prepareNesting()
        bar, btn = dlg._extractWidgets()
        inserted = False
        for i, bar2 in enumerate(self._subBars):
            if bar2.hidden:
                self._subBars.pop(i)
                bar2.hide()
                bar2.setParent(None)
                self._subBars.insert(i, bar)
                inserted = True
                break
        if not inserted:
            self._subBars.append(bar)
        while self.nestedLayout.count() > 0:
            self.nestedLayout.takeAt(0)
        for b in self._subBars:
            self.nestedLayout.addWidget(b)

    def _removeSubDialog(self, dlg):
        bar, btn = dlg._extractWidgets()
        bar.hide()

    def _prepareNesting(self):
        if self._nestingReady is False:
            self._topLayout = QtWidgets.QGridLayout()
            self.setLayout(self._topLayout)
            self._topLayout.setContentsMargins(0, 0, 0, 0)
            self.nestedVBox = QtWidgets.QWidget()
            self._topLayout.addWidget(self.nestedVBox, 0, 0, 1, 2)
            self.nestedLayout = QtWidgets.QVBoxLayout()
            self.nestedVBox.setLayout(self.nestedLayout)
            bar, btn = self._extractWidgets()
            self.nestedLayout.addWidget(bar)
            self._subBars.append(bar)
            self._topLayout.addWidget(btn, 1, 1, 1, 1)
            self._topLayout.setColumnStretch(0, 100)
            self._topLayout.setColumnStretch(1, 1)
            self._topLayout.setRowStretch(0, 100)
            self._topLayout.setRowStretch(1, 1)
            self._nestingReady = True

    def _extractWidgets(self):
        if self._nestableWidgets is None:
            label = [ch for ch in self.children() if isinstance(ch, QtWidgets.QLabel)][0]
            bar = [ch for ch in self.children() if isinstance(ch, QtWidgets.QProgressBar)][0]
            btn = [ch for ch in self.children() if isinstance(ch, QtWidgets.QPushButton)][0]
            sw = ProgressWidget(label, bar)
            self._nestableWidgets = (sw, btn)
        return self._nestableWidgets

    def resizeEvent(self, ev):
        if self._nestingReady:
            return
        return super().resizeEvent(ev)

    def setValue(self, val):
        if self.disabled:
            return
        QtWidgets.QProgressDialog.setValue(self, val)
        if self.windowModality() == QtCore.Qt.WindowModality.WindowModal:
            now = perf_counter()
            if self._lastProcessEvents is None or now - self._lastProcessEvents > 0.2:
                QtWidgets.QApplication.processEvents()
                self._lastProcessEvents = now

    def setLabelText(self, val):
        if self.disabled:
            return
        QtWidgets.QProgressDialog.setLabelText(self, val)

    def setMaximum(self, val):
        if self.disabled:
            return
        QtWidgets.QProgressDialog.setMaximum(self, val)

    def setMinimum(self, val):
        if self.disabled:
            return
        QtWidgets.QProgressDialog.setMinimum(self, val)

    def wasCanceled(self):
        if self.disabled:
            return False
        return QtWidgets.QProgressDialog.wasCanceled(self)

    def maximum(self):
        if self.disabled:
            return 0
        return QtWidgets.QProgressDialog.maximum(self)

    def minimum(self):
        if self.disabled:
            return 0
        return QtWidgets.QProgressDialog.minimum(self)