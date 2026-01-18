import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
class cmd_ancestor_growth(commands.Command):
    """Figure out the ancestor graph for LOCATION"""
    takes_args = ['location?']
    encoding_type = 'replace'
    hidden = True

    def run(self, location='.'):
        try:
            wt = workingtree.WorkingTree.open_containing(location)[0]
        except errors.NoWorkingTree:
            a_branch = branch.Branch.open(location)
            last_rev = a_branch.last_revision()
        else:
            a_branch = wt.branch
            last_rev = wt.last_revision()
        with a_branch.lock_read():
            graph = a_branch.repository.get_graph()
            revno = 0
            cur_parents = 0
            sorted_graph = tsort.merge_sort(graph.iter_ancestry([last_rev]), last_rev)
            for num, node_name, depth, isend in reversed(sorted_graph):
                cur_parents += 1
                if depth == 0:
                    revno += 1
                    self.outf.write('%4d, %4d\n' % (revno, cur_parents))