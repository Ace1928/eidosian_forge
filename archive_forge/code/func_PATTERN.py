@property
def PATTERN(self):
    u"""
        Uses the mapping of names to features to return a PATTERN suitable
        for using the lib2to3 patcomp.
        """
    self.update_mapping()
    return u' |\n'.join([pattern_unformatted % (f.name, f._pattern) for f in iter(self)])