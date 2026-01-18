class IncompatibleSegmentNumberError(IncompatibleGlyphsError):

    def __str__(self):
        return 'Glyphs named %s have different number of segments' % self.combined_name