from cinderclient import base
from cinderclient import shell_utils
class ListExtResource(base.Resource):

    @property
    def summary(self):
        descr = self.description.strip()
        if not descr:
            return '??'
        lines = descr.split('\n')
        if len(lines) == 1:
            return lines[0]
        else:
            return lines[0] + '...'