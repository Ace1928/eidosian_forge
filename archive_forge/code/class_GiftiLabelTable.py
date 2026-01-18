from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
class GiftiLabelTable(xml.XmlSerializable):
    """Gifti label table: a sequence of key, label pairs

    From the gifti spec dated 2011-01-14:
        The label table is used by DataArrays whose values are an key into the
        LabelTable's labels. A file should contain at most one LabelTable and
        it must be located in the file prior to any DataArray elements.
    """

    def __init__(self):
        self.labels = []

    def __repr__(self):
        return f'<GiftiLabelTable {self.labels!r}>'

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def _to_xml_element(self):
        labeltable = xml.Element('LabelTable')
        for ele in self.labels:
            label = xml.SubElement(labeltable, 'Label')
            label.attrib['Key'] = str(ele.key)
            label.text = ele.label
            for attr in ('Red', 'Green', 'Blue', 'Alpha'):
                if getattr(ele, attr.lower(), None) is not None:
                    label.attrib[attr] = str(getattr(ele, attr.lower()))
        return labeltable

    def print_summary(self):
        print(self.get_labels_as_dict())