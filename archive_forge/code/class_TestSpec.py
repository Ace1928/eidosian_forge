import argparse
import collections
import io
import json
import logging
import os
import sys
from xml.etree import ElementTree as ET
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang import parse
class TestSpec(object):
    """Stores agregated ctest specification, including the name, command, and
     any properties that are set.
  """

    def __init__(self, name, argv, cwd):
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.props = {}

    def as_odict(self):
        """Return a dictionary representation of the specification, suitable for
       serialization to JSON.
    """
        props = collections.OrderedDict()
        for key, value in sorted(self.props.items()):
            if key == 'LABELS':
                value = value.split(';')
            if key == 'TIMEOUT':
                value = int(value)
            if key.startswith('_'):
                continue
            props[key.lower()] = value
        out = collections.OrderedDict([('name', self.name), ('argv', self.argv), ('cwd', self.cwd), ('props', props)])
        return out

    def as_element(self):
        """Return an ElementTree representation of the specification, suitable
       for serialialization to XML.
    """
        elem = ET.Element('test')
        elem.set('name', self.name)
        elem.set('cwd', self.cwd)
        argv_elem = ET.Element('argv')
        elem.append(argv_elem)
        for value in self.argv:
            item = ET.Element('arg')
            argv_elem.append(item)
            item.set('value', value)
        for key, value in sorted(self.props.items()):
            if key.startswith('_'):
                continue
            if key == 'LABELS':
                labels_elem = ET.Element('labels')
                elem.append(labels_elem)
                for label in value.split(';'):
                    item = ET.Element('label')
                    labels_elem.append(item)
                    item.set('value', label)
                continue
            elem.set(key.lower(), value)
        return elem