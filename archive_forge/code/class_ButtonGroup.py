import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
class ButtonGroup(object):
    """ Encapsulate the configuration of a button group and its implementation.
    """

    def __init__(self):
        """ Initialise the button group. """
        self.exclusive = True
        self.object = None