import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
class Cifti2Label(xml.XmlSerializable):
    """CIFTI-2 label: association of integer key with a name and RGBA values

    For all color components, value is floating point with range 0.0 to 1.0.

    * Description - Associates a label key value with a name and a display
      color.
    * Attributes

        * Key - Integer, data value which is assigned this name and color.
        * Red - Red color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Green - Green color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Blue - Blue color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Alpha - Alpha color component for label. Value is floating point with
          range 0.0 to 1.0.

    * Child Elements: [NA]
    * Text Content - Name of the label.
    * Parent Element - LabelTable

    Attributes
    ----------
    key : int, optional
        Integer, data value which is assigned this name and color.
    label : str, optional
        Name of the label.
    red : float, optional
        Red color component for label (between 0 and 1).
    green : float, optional
        Green color component for label (between 0 and 1).
    blue : float, optional
        Blue color component for label (between 0 and 1).
    alpha : float, optional
        Alpha color component for label (between 0 and 1).
    """

    def __init__(self, key=0, label='', red=0.0, green=0.0, blue=0.0, alpha=0.0):
        self.key = int(key)
        self.label = str(label)
        self.red = _float_01(red)
        self.green = _float_01(green)
        self.blue = _float_01(blue)
        self.alpha = _float_01(alpha)

    @property
    def rgba(self):
        """Returns RGBA as tuple"""
        return (self.red, self.green, self.blue, self.alpha)

    def _to_xml_element(self):
        if self.label == '':
            raise Cifti2HeaderError('Label needs a name')
        try:
            v = int(self.key)
        except ValueError:
            raise Cifti2HeaderError('The key must be an integer')
        for c_ in ('red', 'blue', 'green', 'alpha'):
            try:
                v = _float_01(getattr(self, c_))
            except ValueError:
                raise Cifti2HeaderError(f'Label invalid {c_} needs to be a float between 0 and 1. and it is {v}')
        lab = xml.Element('Label')
        lab.attrib['Key'] = str(self.key)
        lab.text = str(self.label)
        for name in ('red', 'green', 'blue', 'alpha'):
            val = getattr(self, name)
            attr = '0' if val == 0 else '1' if val == 1 else str(val)
            lab.attrib[name.capitalize()] = attr
        return lab