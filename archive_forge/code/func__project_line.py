import Bio.GenBank
def _project_line(self):
    output = ''
    if len(self.projects) > 0:
        output = Record.BASE_FORMAT % 'PROJECT'
        output += f'{'  '.join(self.projects)}\n'
    return output