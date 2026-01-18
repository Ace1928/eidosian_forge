from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
class AddCustomIDAction(add.AddAction):

    def __call__(self, inv, parent_ie, path, kind):
        file_id = (kind + '-' + path.replace('/', '%')).encode('utf-8')
        if self.should_print:
            self._to_file.write('added %s with id %s\n' % (path, file_id.decode('utf-8')))
        return file_id