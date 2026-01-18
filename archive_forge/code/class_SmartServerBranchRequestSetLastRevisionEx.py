import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestSetLastRevisionEx(SmartServerSetTipRequest):

    def do_tip_change_with_locked_branch(self, branch, new_last_revision_id, allow_divergence, allow_overwrite_descendant):
        """Set the last revision of the branch.

        New in 1.6.

        :param new_last_revision_id: the revision ID to set as the last
            revision of the branch.
        :param allow_divergence: A flag.  If non-zero, change the revision ID
            even if the new_last_revision_id's ancestry has diverged from the
            current last revision.  If zero, a 'Diverged' error will be
            returned if new_last_revision_id is not a descendant of the current
            last revision.
        :param allow_overwrite_descendant:  A flag.  If zero and
            new_last_revision_id is not a descendant of the current last
            revision, then the last revision will not be changed.  If non-zero
            and there is no divergence, then the last revision is always
            changed.

        :returns: on success, a tuple of ('ok', revno, revision_id), where
            revno and revision_id are the new values of the current last
            revision info.  The revision_id might be different to the
            new_last_revision_id if allow_overwrite_descendant was not set.
        """
        do_not_overwrite_descendant = not allow_overwrite_descendant
        try:
            last_revno, last_rev = branch.last_revision_info()
            graph = branch.repository.get_graph()
            if not allow_divergence or do_not_overwrite_descendant:
                relation = branch._revision_relations(last_rev, new_last_revision_id, graph)
                if relation == 'diverged' and (not allow_divergence):
                    return FailedSmartServerResponse((b'Diverged',))
                if relation == 'a_descends_from_b' and do_not_overwrite_descendant:
                    return SuccessfulSmartServerResponse((b'ok', last_revno, last_rev))
            new_revno = graph.find_distance_to_null(new_last_revision_id, [(last_rev, last_revno)])
            branch.set_last_revision_info(new_revno, new_last_revision_id)
        except errors.GhostRevisionsHaveNoRevno:
            return FailedSmartServerResponse((b'NoSuchRevision', new_last_revision_id))
        return SuccessfulSmartServerResponse((b'ok', new_revno, new_last_revision_id))