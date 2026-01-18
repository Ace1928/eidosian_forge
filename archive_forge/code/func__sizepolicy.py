import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _sizepolicy(self, prop, widget):
    values = [int(child.text) for child in prop]
    if len(values) == 2:
        horstretch, verstretch = values
        hsizetype = getattr(QtWidgets.QSizePolicy, prop.get('hsizetype'))
        vsizetype = getattr(QtWidgets.QSizePolicy, prop.get('vsizetype'))
    else:
        hsizetype, vsizetype, horstretch, verstretch = values
        hsizetype = QtWidgets.QSizePolicy.Policy(hsizetype)
        vsizetype = QtWidgets.QSizePolicy.Policy(vsizetype)
    sizePolicy = self.factory.createQObject('QSizePolicy', 'sizePolicy', (hsizetype, vsizetype), is_attribute=False)
    sizePolicy.setHorizontalStretch(horstretch)
    sizePolicy.setVerticalStretch(verstretch)
    sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy