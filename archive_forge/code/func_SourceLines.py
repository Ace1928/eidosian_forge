import threading
from tensorboard import errors
def SourceLines(self, run, index):
    runs = self.Runs()
    if run not in runs:
        return None
    try:
        host_name, file_path = self._reader.source_file_list()[index]
    except IndexError:
        raise errors.NotFoundError('There is no source-code file at index %d' % index)
    return {'host_name': host_name, 'file_path': file_path, 'lines': self._reader.source_lines(host_name, file_path)}