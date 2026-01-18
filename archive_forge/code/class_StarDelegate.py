from PySide2.QtWidgets import (QItemDelegate, QStyledItemDelegate, QStyle)
from starrating import StarRating
from stareditor import StarEditor
class StarDelegate(QStyledItemDelegate):
    """ A subclass of QStyledItemDelegate that allows us to render our
        pretty star ratings.
    """

    def __init__(self, parent=None):
        super(StarDelegate, self).__init__(parent)

    def paint(self, painter, option, index):
        """ Paint the items in the table.

            If the item referred to by <index> is a StarRating, we handle the
            painting ourselves. For the other items, we let the base class
            handle the painting as usual.

            In a polished application, we'd use a better check than the
            column number to find out if we needed to paint the stars, but
            it works for the purposes of this example.
        """
        if index.column() == 3:
            starRating = StarRating(index.data())
            if option.state & QStyle.State_Selected:
                painter.fillRect(option.rect, painter.brush())
            starRating.paint(painter, option.rect, option.palette)
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

    def sizeHint(self, option, index):
        """ Returns the size needed to display the item in a QSize object. """
        if index.column() == 3:
            starRating = StarRating(index.data())
            return starRating.sizeHint()
        else:
            return QStyledItemDelegate.sizeHint(self, option, index)

    def createEditor(self, parent, option, index):
        """ Creates and returns the custom StarEditor object we'll use to edit
            the StarRating.
        """
        if index.column() == 3:
            editor = StarEditor(parent)
            editor.editingFinished.connect(self.commitAndCloseEditor)
            return editor
        else:
            return QStyledItemDelegate.createEditor(self, parent, option, index)

    def setEditorData(self, editor, index):
        """ Sets the data to be displayed and edited by our custom editor. """
        if index.column() == 3:
            editor.starRating = StarRating(index.data())
        else:
            QStyledItemDelegate.setEditorData(self, editor, index)

    def setModelData(self, editor, model, index):
        """ Get the data from our custom editor and stuffs it into the model.
        """
        if index.column() == 3:
            model.setData(index, editor.starRating.starCount)
        else:
            QStyledItemDelegate.setModelData(self, editor, model, index)

    def commitAndCloseEditor(self):
        """ Erm... commits the data and closes the editor. :) """
        editor = self.sender()
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QStyledItemDelegate.NoHint)