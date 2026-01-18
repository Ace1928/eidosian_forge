import re
from io import StringIO
from Bio.Phylo import Newick
def _info_factory(self, plain, confidence_as_branch_length, branch_length_only, max_confidence, format_confidence, format_branch_length):
    """Return a function that creates a nicely formatted node tag (PRIVATE)."""
    if plain:

        def make_info_string(clade, terminal=False):
            return _get_comment(clade)
    elif confidence_as_branch_length:

        def make_info_string(clade, terminal=False):
            if terminal:
                return ':' + format_confidence % max_confidence + _get_comment(clade)
            else:
                return ':' + format_confidence % clade.confidence + _get_comment(clade)
    elif branch_length_only:

        def make_info_string(clade, terminal=False):
            return ':' + format_branch_length % clade.branch_length + _get_comment(clade)
    else:

        def make_info_string(clade, terminal=False):
            if terminal or not hasattr(clade, 'confidence') or clade.confidence is None:
                return (':' + format_branch_length) % (clade.branch_length or 0.0) + _get_comment(clade)
            else:
                return (format_confidence + ':' + format_branch_length) % (clade.confidence, clade.branch_length or 0.0) + _get_comment(clade)
    return make_info_string