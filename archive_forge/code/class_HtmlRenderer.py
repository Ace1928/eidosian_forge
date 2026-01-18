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
class HtmlRenderer(MimetypeRenderer):
    """
    Base class for all HTML mime type renderers

    mime type: 'text/html'
    """

    def __init__(self, connected=False, full_html=False, requirejs=True, global_init=False, config=None, auto_play=False, post_script=None, animation_opts=None):
        self.config = dict(config) if config else {}
        self.auto_play = auto_play
        self.connected = connected
        self.global_init = global_init
        self.requirejs = requirejs
        self.full_html = full_html
        self.animation_opts = animation_opts
        self.post_script = post_script

    def activate(self):
        if self.global_init:
            if not ipython_display:
                raise ValueError('The {cls} class requires ipython but it is not installed'.format(cls=self.__class__.__name__))
            if not self.requirejs:
                raise ValueError('global_init is only supported with requirejs=True')
            if self.connected:
                script = '        <script type="text/javascript">\n        {win_config}\n        {mathjax_config}\n        if (typeof require !== \'undefined\') {{\n        require.undef("plotly");\n        requirejs.config({{\n            paths: {{\n                \'plotly\': [\'{plotly_cdn}\']\n            }}\n        }});\n        require([\'plotly\'], function(Plotly) {{\n            window._Plotly = Plotly;\n        }});\n        }}\n        </script>\n        '.format(win_config=_window_plotly_config, mathjax_config=_mathjax_config, plotly_cdn=plotly_cdn_url().rstrip('.js'))
            else:
                script = '        <script type="text/javascript">\n        {win_config}\n        {mathjax_config}\n        if (typeof require !== \'undefined\') {{\n        require.undef("plotly");\n        define(\'plotly\', function(require, exports, module) {{\n            {script}\n        }});\n        require([\'plotly\'], function(Plotly) {{\n            window._Plotly = Plotly;\n        }});\n        }}\n        </script>\n        '.format(script=get_plotlyjs(), win_config=_window_plotly_config, mathjax_config=_mathjax_config)
            ipython_display.display_html(script, raw=True)

    def to_mimebundle(self, fig_dict):
        from plotly.io import to_html
        if self.requirejs:
            include_plotlyjs = 'require'
            include_mathjax = False
        elif self.connected:
            include_plotlyjs = 'cdn'
            include_mathjax = 'cdn'
        else:
            include_plotlyjs = True
            include_mathjax = 'cdn'
        post_script = ["\nvar gd = document.getElementById('{plot_id}');\nvar x = new MutationObserver(function (mutations, observer) {{\n        var display = window.getComputedStyle(gd).display;\n        if (!display || display === 'none') {{\n            console.log([gd, 'removed!']);\n            Plotly.purge(gd);\n            observer.disconnect();\n        }}\n}});\n\n// Listen for the removal of the full notebook cells\nvar notebookContainer = gd.closest('#notebook-container');\nif (notebookContainer) {{\n    x.observe(notebookContainer, {childList: true});\n}}\n\n// Listen for the clearing of the current output cell\nvar outputEl = gd.closest('.output');\nif (outputEl) {{\n    x.observe(outputEl, {childList: true});\n}}\n"]
        if self.post_script:
            if not isinstance(self.post_script, (list, tuple)):
                post_script.append(self.post_script)
            else:
                post_script.extend(self.post_script)
        html = to_html(fig_dict, config=self.config, auto_play=self.auto_play, include_plotlyjs=include_plotlyjs, include_mathjax=include_mathjax, post_script=post_script, full_html=self.full_html, animation_opts=self.animation_opts, default_width='100%', default_height=525, validate=False)
        return {'text/html': html}