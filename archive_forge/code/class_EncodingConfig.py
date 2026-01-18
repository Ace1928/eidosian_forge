from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class EncodingConfig(ConfigObject):
    """Options affecting file encoding"""
    _field_registry = []
    emit_byteorder_mark = FieldDescriptor(False, 'If true, emit the unicode byte-order mark (BOM) at the start of the file')
    input_encoding = FieldDescriptor('utf-8', 'Specify the encoding of the input file. Defaults to utf-8')
    output_encoding = FieldDescriptor('utf-8', 'Specify the encoding of the output file. Defaults to utf-8. Note that cmake only claims to support utf-8 so be careful when using anything else')