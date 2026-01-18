import Bio.GenBank
def _version_line(self):
    """Output for the VERSION line (PRIVATE)."""
    if self.version:
        output = Record.BASE_FORMAT % 'VERSION'
        output += self.version
        output += '  GI:'
        output += f'{self.gi}\n'
    else:
        output = ''
    return output