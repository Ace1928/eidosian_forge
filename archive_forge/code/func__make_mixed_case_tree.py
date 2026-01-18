import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def _make_mixed_case_tree(self):
    """Make a working tree with mixed-case filenames."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['CamelCaseParent/', 'lowercaseparent/'])
    self.build_tree_contents([('CamelCaseParent/CamelCase', b'camel case'), ('lowercaseparent/lowercase', b'lower case'), ('lowercaseparent/mixedCase', b'mixedCasecase')])
    return wt