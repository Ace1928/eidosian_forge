import Bio.GenBank
def _features_line(self):
    """Output for the FEATURES line (PRIVATE)."""
    output = ''
    if len(self.features) > 0:
        output += Record.BASE_FEATURE_FORMAT % 'FEATURES'
        output += 'Location/Qualifiers\n'
    return output