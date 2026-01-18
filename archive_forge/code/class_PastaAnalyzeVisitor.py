import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class PastaAnalyzeVisitor(_PastaEditVisitor):
    """AST Visitor that looks for specific API usage without editing anything.

  This is used before any rewriting is done to detect if any symbols are used
  that require changing imports or disabling rewriting altogether.
  """

    def __init__(self, api_analysis_spec):
        super(PastaAnalyzeVisitor, self).__init__(NoUpdateSpec())
        self._api_analysis_spec = api_analysis_spec
        self._results = []

    @property
    def results(self):
        return self._results

    def add_result(self, analysis_result):
        self._results.append(analysis_result)

    def visit_Attribute(self, node):
        """Handle bare Attributes i.e. [tf.foo, tf.bar]."""
        full_name = self._get_full_name(node)
        if full_name:
            detection = self._api_analysis_spec.symbols_to_detect.get(full_name, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)

    def visit_Import(self, node):
        """Handle visiting an import node in the AST.

    Args:
      node: Current Node
    """
        for import_alias in node.names:
            full_import = (import_alias.name, import_alias.asname)
            detection = self._api_analysis_spec.imports_to_detect.get(full_import, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle visiting an import-from node in the AST.

    Args:
      node: Current Node
    """
        if not node.module:
            self.generic_visit(node)
            return
        from_import = node.module
        for import_alias in node.names:
            full_module_name = '%s.%s' % (from_import, import_alias.name)
            full_import = (full_module_name, import_alias.asname)
            detection = self._api_analysis_spec.imports_to_detect.get(full_import, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)