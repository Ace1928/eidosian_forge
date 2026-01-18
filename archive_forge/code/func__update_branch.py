from operator import itemgetter
from breezy import controldir
from ... import errors, osutils, transport
from ...trace import note, show_error
from .helpers import best_format_for_objects_in_a_repository, single_plural
def _update_branch(self, br, last_mark):
    """Update a branch with last revision and tag information.

        :return: whether the branch was changed or not
        """
    last_rev_id = self.cache_mgr.lookup_committish(last_mark)
    with self.repo.lock_read():
        graph = self.repo.get_graph()
        revno = graph.find_distance_to_null(last_rev_id, [])
    existing_revno, existing_last_rev_id = br.last_revision_info()
    changed = False
    if revno != existing_revno or last_rev_id != existing_last_rev_id:
        br.set_last_revision_info(revno, last_rev_id)
        changed = True
    my_tags = {}
    if self.tags:
        graph = self.repo.get_graph()
        ancestry = [r for r, ps in graph.iter_ancestry([last_rev_id]) if ps is not None]
        for tag, rev in self.tags.items():
            if rev in ancestry:
                my_tags[tag] = rev
        if my_tags:
            br.tags._set_tag_dict(my_tags)
            changed = True
    if changed:
        tagno = len(my_tags)
        note('\t branch %s now has %d %s and %d %s', br.nick, revno, single_plural(revno, 'revision', 'revisions'), tagno, single_plural(tagno, 'tag', 'tags'))
    return changed