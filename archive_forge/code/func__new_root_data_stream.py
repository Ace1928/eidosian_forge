import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def _new_root_data_stream(root_keys_to_create, rev_id_to_root_id_map, parent_map, repo, graph=None):
    """Generate a texts substream of synthesised root entries.

    Used in fetches that do rich-root upgrades.

    Args:
      root_keys_to_create: iterable of (root_id, rev_id) pairs describing
        the root entries to create.
      rev_id_to_root_id_map: dict of known rev_id -> root_id mappings for
        calculating the parents.  If a parent rev_id is not found here then it
        will be recalculated.
      parent_map: a parent map for all the revisions in
        root_keys_to_create.
      graph: a graph to use instead of repo.get_graph().
    """
    from .versionedfile import ChunkedContentFactory
    for root_key in root_keys_to_create:
        root_id, rev_id = root_key
        parent_keys = _parent_keys_for_root_version(root_id, rev_id, rev_id_to_root_id_map, parent_map, repo, graph)
        yield ChunkedContentFactory(root_key, parent_keys, None, [])