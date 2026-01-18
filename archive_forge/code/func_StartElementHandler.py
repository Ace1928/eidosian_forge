import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError
import numpy as np
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
def StartElementHandler(self, name, attrs):
    self.flush_chardata()
    if self.verbose > 0:
        print('Start element:\n\t', repr(name), attrs)
    if name == 'GIFTI':
        self.img = GiftiImage()
        if 'Version' in attrs:
            self.img.version = attrs['Version']
        if 'NumberOfDataArrays' in attrs:
            self.expected_numDA = int(attrs['NumberOfDataArrays'])
        self.fsm_state.append('GIFTI')
    elif name == 'MetaData':
        self.fsm_state.append('MetaData')
        if len(self.fsm_state) == 2:
            self.meta_global = GiftiMetaData()
        else:
            self.meta_da = GiftiMetaData()
    elif name == 'MD':
        self.nvpair = ['', '']
        self.fsm_state.append('MD')
    elif name == 'Name':
        if self.nvpair is None:
            raise GiftiParseError
        self.write_to = 'Name'
    elif name == 'Value':
        if self.nvpair is None:
            raise GiftiParseError
        self.write_to = 'Value'
    elif name == 'LabelTable':
        self.lata = GiftiLabelTable()
        self.fsm_state.append('LabelTable')
    elif name == 'Label':
        self.label = GiftiLabel()
        if 'Index' in attrs:
            self.label.key = int(attrs['Index'])
        if 'Key' in attrs:
            self.label.key = int(attrs['Key'])
        if 'Red' in attrs:
            self.label.red = float(attrs['Red'])
        if 'Green' in attrs:
            self.label.green = float(attrs['Green'])
        if 'Blue' in attrs:
            self.label.blue = float(attrs['Blue'])
        if 'Alpha' in attrs:
            self.label.alpha = float(attrs['Alpha'])
        self.write_to = 'Label'
    elif name == 'DataArray':
        self.da = GiftiDataArray()
        if 'Intent' in attrs:
            self.da.intent = intent_codes.code[attrs['Intent']]
        if 'DataType' in attrs:
            self.da.datatype = data_type_codes.code[attrs['DataType']]
        if 'ArrayIndexingOrder' in attrs:
            self.da.ind_ord = array_index_order_codes.code[attrs['ArrayIndexingOrder']]
        num_dim = int(attrs.get('Dimensionality', 0))
        for i in range(num_dim):
            di = f'Dim{i}'
            if di in attrs:
                self.da.dims.append(int(attrs[di]))
        assert len(self.da.dims) == num_dim
        if 'Encoding' in attrs:
            self.da.encoding = gifti_encoding_codes.code[attrs['Encoding']]
        if 'Endian' in attrs:
            self.da.endian = gifti_endian_codes.code[attrs['Endian']]
        if 'ExternalFileName' in attrs:
            self.da.ext_fname = attrs['ExternalFileName']
        if 'ExternalFileOffset' in attrs:
            self.da.ext_offset = _str2int(attrs['ExternalFileOffset'])
        self.img.darrays.append(self.da)
        self.fsm_state.append('DataArray')
    elif name == 'CoordinateSystemTransformMatrix':
        self.coordsys = GiftiCoordSystem()
        self.img.darrays[-1].coordsys = self.coordsys
        self.fsm_state.append('CoordinateSystemTransformMatrix')
    elif name == 'DataSpace':
        if self.coordsys is None:
            raise GiftiParseError
        self.write_to = 'DataSpace'
    elif name == 'TransformedSpace':
        if self.coordsys is None:
            raise GiftiParseError
        self.write_to = 'TransformedSpace'
    elif name == 'MatrixData':
        if self.coordsys is None:
            raise GiftiParseError
        self.write_to = 'MatrixData'
    elif name == 'Data':
        self.write_to = 'Data'