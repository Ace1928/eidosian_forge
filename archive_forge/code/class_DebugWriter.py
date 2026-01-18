from pprint import pprint
from .base import WriterBase
class DebugWriter(WriterBase):
    """Consumes output from nbconvert export...() methods and writes useful
    debugging information to the stdout.  The information includes a list of
    resources that were extracted from the notebook(s) during export."""

    def write(self, output, resources, notebook_name='notebook', **kw):
        """
        Consume and write Jinja output.

        See base for more...
        """
        if isinstance(resources['outputs'], dict):
            print('outputs extracted from %s' % notebook_name)
            print('-' * 80)
            pprint(resources['outputs'], indent=2, width=70)
        else:
            print('no outputs extracted from %s' % notebook_name)
        print('=' * 80)