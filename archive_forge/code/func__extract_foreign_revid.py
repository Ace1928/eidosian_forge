from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def _extract_foreign_revid(rev):
    try:
        foreign_revid, mapping = foreign.foreign_vcs_registry.parse_revision_id(rev.revision_id)
    except errors.InvalidRevisionId:
        pass
    else:
        yield (mapping.vcs.abbreviation, mapping.vcs.serialize_foreign_revid(foreign_revid))