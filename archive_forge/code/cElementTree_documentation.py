from __future__ import absolute_import
import warnings
from .common import _generate_etree_functions
from xml.etree.cElementTree import TreeBuilder as _TreeBuilder
from xml.etree.cElementTree import parse as _parse
from xml.etree.cElementTree import tostring
from xml.etree.ElementTree import iterparse as _iterparse
from .ElementTree import (
Defused xml.etree.cElementTree
