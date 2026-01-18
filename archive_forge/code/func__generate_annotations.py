from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def _generate_annotations(self, rootwriter, result, options):
    self.annotate(rootwriter)
    coverage_xml_filename = Options.annotate_coverage_xml or options.annotate_coverage_xml
    if coverage_xml_filename and os.path.exists(coverage_xml_filename):
        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET
        coverage_xml = ET.parse(coverage_xml_filename).getroot()
        for el in coverage_xml.iter():
            el.tail = None
    else:
        coverage_xml = None
    rootwriter.save_annotation(result.main_source_file, result.c_file, coverage_xml=coverage_xml)
    if not self.scope.included_files:
        return
    search_include_file = self.scope.context.search_include_directories
    target_dir = os.path.abspath(os.path.dirname(result.c_file))
    for included_file in self.scope.included_files:
        target_file = os.path.abspath(os.path.join(target_dir, included_file))
        target_file_dir = os.path.dirname(target_file)
        if not target_file_dir.startswith(target_dir):
            continue
        source_file = search_include_file(included_file, source_pos=self.pos, include=True)
        if not source_file:
            continue
        if target_file_dir != target_dir and (not os.path.exists(target_file_dir)):
            try:
                os.makedirs(target_file_dir)
            except OSError as e:
                import errno
                if e.errno != errno.EEXIST:
                    raise
        rootwriter.save_annotation(source_file, target_file, coverage_xml=coverage_xml)