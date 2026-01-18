import unittest
from nbformat import v4 as nbformat
class NBClientTestsBase(unittest.TestCase):

    def build_notebook(self, with_json_outputs=False):
        """Build a notebook in memory for use with NotebookClient tests"""
        outputs = [nbformat.new_output('stream', name='stdout', text='a'), nbformat.new_output('display_data', data={'text/plain': 'b'}), nbformat.new_output('stream', name='stdout', text='c'), nbformat.new_output('stream', name='stdout', text='d'), nbformat.new_output('stream', name='stderr', text='e'), nbformat.new_output('stream', name='stderr', text='f'), nbformat.new_output('display_data', data={'image/png': 'Zw=='}), nbformat.new_output('display_data', data={'application/pdf': 'aA=='})]
        if with_json_outputs:
            outputs.extend([nbformat.new_output('display_data', data={'application/json': [1, 2, 3]}), nbformat.new_output('display_data', data={'application/json': {'a': 1, 'c': {'b': 2}}}), nbformat.new_output('display_data', data={'application/json': 'abc'}), nbformat.new_output('display_data', data={'application/json': 15.03})])
        cells = [nbformat.new_code_cell(source='$ e $', execution_count=1, outputs=outputs), nbformat.new_markdown_cell(source='$ e $')]
        return nbformat.new_notebook(cells=cells)

    def build_resources(self):
        """Build an empty resources dictionary."""
        return {'metadata': {}}

    @classmethod
    def merge_dicts(cls, *dict_args):
        outcome = {}
        for d in dict_args:
            outcome.update(d)
        return outcome