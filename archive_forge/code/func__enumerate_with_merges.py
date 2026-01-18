from . import errors, log
def _enumerate_with_merges(branch, ancestry, graph, tip_revno, tip, backward=True):
    """Enumerate the revisions for the ancestry.

    :param branch: The branch we care about
    :param ancestry: A set of revisions that we care about
    :param graph: A Graph which lets us find the parents for a revision
    :param tip_revno: The revision number for the tip revision
    :param tip: The tip of the ancsetry
    :param backward: Show oldest versions first when True, newest versions
        first when False.
    :return: [(revno, revision_id)] for all revisions in ancestry that
        are parents from tip, or None if ancestry is None.
    """
    if ancestry is None:
        return None
    if not ancestry:
        return []
    merge_sorted_revisions = branch.iter_merge_sorted_revisions()
    merge_sorted_revisions = [(0, revid, n, d, e) for revid, n, d, e in merge_sorted_revisions if revid in ancestry]
    if not backward:
        merge_sorted_revisions = log.reverse_by_depth(merge_sorted_revisions)
    revline = []
    for seq, rev_id, merge_depth, revno, end_of_merge in merge_sorted_revisions:
        revline.append(('.'.join(map(str, revno)), rev_id, merge_depth))
    return revline