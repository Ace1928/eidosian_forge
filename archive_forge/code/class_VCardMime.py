import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
class VCardMime(QtMacExtras.QMacPasteboardMime):

    def __init__(self, t=QtMacExtras.QMacPasteboardMime.MIME_ALL):
        super(VCardMime, self).__init__(t)

    def convertorName(self):
        return 'VCardMime'

    def canConvert(self, mime, flav):
        if self.mimeFor(flav) == mime:
            return True
        else:
            return False

    def mimeFor(self, flav):
        if flav == 'public.vcard':
            return 'application/x-mycompany-VCard'
        else:
            return ''

    def flavorFor(self, mime):
        if mime == 'application/x-mycompany-VCard':
            return 'public.vcard'
        else:
            return ''

    def convertToMime(self, mime, data, flav):
        all = QtCore.QByteArray()
        for i in data:
            all += i
        return all

    def convertFromMime(mime, data, flav):
        return []