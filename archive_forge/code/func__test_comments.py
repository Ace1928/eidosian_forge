def _test_comments():
    """A self-test on comment processing.  Returns number of test failures."""

    def _testrm(a, b, collapse):
        b2 = remove_comments(a, collapse)
        if b != b2:
            print('Comment test failed:')
            print('   remove_comments( %s, collapse_spaces=%s ) -> %s' % (repr(a), repr(collapse), repr(b2)))
            print('   expected %s' % repr(b))
            return 1
        return 0
    failures = 0
    failures += _testrm('', '', False)
    failures += _testrm('(hello)', '', False)
    failures += _testrm('abc (hello) def', 'abc  def', False)
    failures += _testrm('abc (he(xyz)llo) def', 'abc  def', False)
    failures += _testrm('abc (he\\(xyz)llo) def', 'abc llo) def', False)
    failures += _testrm('abc(hello)def', 'abcdef', True)
    failures += _testrm('abc (hello) def', 'abc def', True)
    failures += _testrm('abc   (hello)def', 'abc def', True)
    failures += _testrm('abc(hello)  def', 'abc def', True)
    failures += _testrm('abc(hello) (world)def', 'abc def', True)
    failures += _testrm('abc(hello)(world)def', 'abcdef', True)
    failures += _testrm('  (hello) (world) def', 'def', True)
    failures += _testrm('abc  (hello) (world) ', 'abc', True)
    return failures