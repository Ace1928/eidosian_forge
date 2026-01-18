import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
@staticmethod
def _form_layout_role(layout_position):
    if layout_position[3] > 1:
        role = QtWidgets.QFormLayout.SpanningRole
    elif layout_position[1] == 1:
        role = QtWidgets.QFormLayout.FieldRole
    else:
        role = QtWidgets.QFormLayout.LabelRole
    return role