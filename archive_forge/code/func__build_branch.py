from breezy import branch, tests
def _build_branch(self):
    tree = self.make_branch_and_tree('test')
    with open('test/foo', 'wb') as f:
        f.write(b'1111\n')
    tree.add('foo')
    tree.commit('added foo', rev_id=b'revision_1')
    with open('test/foo', 'wb') as f:
        f.write(b'2222\n')
    tree.commit('updated foo', rev_id=b'revision_2')
    with open('test/foo', 'wb') as f:
        f.write(b'3333\n')
    tree.commit('updated foo again', rev_id=b'revision_3')
    return tree