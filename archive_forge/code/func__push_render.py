import time
def _push_render(self):
    """Render the plot with bokeh.io and push to notebook.
        """
    bokeh.io.push_notebook(handle=self.handle)
    self.last_update = time.time()