class Merge2(TextMerge):
    """ Two-way merge.
    In a two way merge, common regions are shown as unconflicting, and uncommon
    regions produce conflicts.
    """

    def __init__(self, lines_a, lines_b, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER, split_marker=TextMerge.SPLIT_MARKER):
        TextMerge.__init__(self, a_marker, b_marker, split_marker)
        self.lines_a = lines_a
        self.lines_b = lines_b

    def _merge_struct(self):
        """Return structured merge info.
        See TextMerge docstring.
        """
        import patiencediff
        sm = patiencediff.PatienceSequenceMatcher(None, self.lines_a, self.lines_b)
        pos_a = 0
        pos_b = 0
        for ai, bi, l in sm.get_matching_blocks():
            yield (self.lines_a[pos_a:ai], self.lines_b[pos_b:bi])
            yield (self.lines_a[ai:ai + l],)
            pos_a = ai + l
            pos_b = bi + l
        yield (self.lines_a[pos_a:-1], self.lines_b[pos_b:-1])