import gradio as gr
from gradio.routes import App
from gradio.utils import BaseReloader
class JupyterReloader(BaseReloader):
    """Swap a running blocks class in a notebook with the latest cell contents."""

    def __init__(self, ipython) -> None:
        super().__init__()
        self._cell_tracker = CellIdTracker(ipython)
        self._running: dict[str, gr.Blocks] = {}

    @property
    def current_cell(self):
        return self._cell_tracker.current_cell

    @property
    def running_app(self) -> App:
        if not self.running_demo.server:
            raise RuntimeError('Server not running')
        return self.running_demo.server.running_app

    @property
    def running_demo(self):
        return self._running[self.current_cell]

    def demo_tracked(self) -> bool:
        return self.current_cell in self._running

    def track(self, demo: gr.Blocks):
        self._running[self.current_cell] = demo