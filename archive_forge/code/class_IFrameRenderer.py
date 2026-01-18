import base64
import json
import webbrowser
import inspect
import os
from os.path import isdir
from plotly import utils, optional_imports
from plotly.io import to_json, to_image, write_image, write_html
from plotly.io._orca import ensure_server
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import _get_jconfig, get_plotlyjs
from plotly.tools import return_figure_from_figure_or_data
class IFrameRenderer(MimetypeRenderer):
    """
    Renderer to display interactive figures using an IFrame.  HTML
    representations of Figures are saved to an `iframe_figures/` directory and
    iframe HTML elements that reference these files are inserted into the
    notebook.

    With this approach, neither plotly.js nor the figure data are embedded in
    the notebook, so this is a good choice for notebooks that contain so many
    large figures that basic operations (like saving and opening) become
    very slow.

    Notebooks using this renderer will display properly when exported to HTML
    as long as the `iframe_figures/` directory is placed in the same directory
    as the exported html file.

    Note that the HTML files in `iframe_figures/` are numbered according to
    the IPython cell execution count and so they will start being overwritten
    each time the kernel is restarted.  This directory may be deleted whenever
    the kernel is restarted and it will be automatically recreated.

    mime type: 'text/html'
    """

    def __init__(self, config=None, auto_play=False, post_script=None, animation_opts=None, include_plotlyjs=True, html_directory='iframe_figures'):
        self.config = config
        self.auto_play = auto_play
        self.post_script = post_script
        self.animation_opts = animation_opts
        self.include_plotlyjs = include_plotlyjs
        self.html_directory = html_directory

    def to_mimebundle(self, fig_dict):
        from plotly.io import write_html
        iframe_buffer = 20
        layout = fig_dict.get('layout', {})
        if layout.get('width', False):
            iframe_width = str(layout['width'] + iframe_buffer) + 'px'
        else:
            iframe_width = '100%'
        if layout.get('height', False):
            iframe_height = layout['height'] + iframe_buffer
        else:
            iframe_height = str(525 + iframe_buffer) + 'px'
        filename = self.build_filename()
        try:
            os.makedirs(self.html_directory)
        except OSError as error:
            if not isdir(self.html_directory):
                raise
        write_html(fig_dict, filename, config=self.config, auto_play=self.auto_play, include_plotlyjs=self.include_plotlyjs, include_mathjax='cdn', auto_open=False, post_script=self.post_script, animation_opts=self.animation_opts, default_width='100%', default_height=525, validate=False)
        iframe_html = '<iframe\n    scrolling="no"\n    width="{width}"\n    height="{height}"\n    src="{src}"\n    frameborder="0"\n    allowfullscreen\n></iframe>\n'.format(width=iframe_width, height=iframe_height, src=self.build_url(filename))
        return {'text/html': iframe_html}

    def build_filename(self):
        ip = IPython.get_ipython() if IPython else None
        try:
            cell_number = list(ip.history_manager.get_tail(1))[0][1] + 1 if ip else 0
        except Exception:
            cell_number = 0
        return '{dirname}/figure_{cell_number}.html'.format(dirname=self.html_directory, cell_number=cell_number)

    def build_url(self, filename):
        return filename