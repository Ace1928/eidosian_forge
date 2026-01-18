import gradio as gr
from gradio.routes import App
from gradio.utils import BaseReloader
class CellIdTracker:
    """Determines the most recently run cell in the notebook.

    Needed to keep track of which demo the user is updating.
    """

    def __init__(self, ipython):
        ipython.events.register('pre_run_cell', self.pre_run_cell)
        self.shell = ipython
        self.current_cell: str = ''

    def pre_run_cell(self, info):
        self._current_cell = info.cell_id