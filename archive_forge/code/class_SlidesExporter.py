from copy import deepcopy
from warnings import warn
from traitlets import Bool, Unicode, default
from nbconvert.preprocessors.base import Preprocessor
from .html import HTMLExporter
class SlidesExporter(HTMLExporter):
    """Exports HTML slides with reveal.js"""
    export_from_notebook = 'Reveal.js slides'

    @default('template_name')
    def _template_name_default(self):
        return 'reveal'

    @default('file_extension')
    def _file_extension_default(self):
        return '.slides.html'

    @default('template_extension')
    def _template_extension_default(self):
        return '.html.j2'
    reveal_url_prefix = Unicode(help='The URL prefix for reveal.js (version 3.x).\n        This defaults to the reveal CDN, but can be any url pointing to a copy\n        of reveal.js.\n\n        For speaker notes to work, this must be a relative path to a local\n        copy of reveal.js: e.g., "reveal.js".\n\n        If a relative path is given, it must be a subdirectory of the\n        current directory (from which the server is run).\n\n        See the usage documentation\n        (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)\n        for more details.\n        ').tag(config=True)

    @default('reveal_url_prefix')
    def _reveal_url_prefix_default(self):
        if 'RevealHelpPreprocessor.url_prefix' in self.config:
            warn('Please update RevealHelpPreprocessor.url_prefix to SlidesExporter.reveal_url_prefix in config files.', stacklevel=2)
            return self.config.RevealHelpPreprocessor.url_prefix
        return 'https://unpkg.com/reveal.js@4.0.2'
    reveal_theme = Unicode('simple', help='\n        Name of the reveal.js theme to use.\n\n        We look for a file with this name under\n        ``reveal_url_prefix``/css/theme/``reveal_theme``.css.\n\n        https://github.com/hakimel/reveal.js/tree/master/css/theme has\n        list of themes that ship by default with reveal.js.\n        ').tag(config=True)
    reveal_transition = Unicode('slide', help='\n        Name of the reveal.js transition to use.\n\n        The list of transitions that ships by default with reveal.js are:\n        none, fade, slide, convex, concave and zoom.\n        ').tag(config=True)
    reveal_scroll = Bool(False, help='\n        If True, enable scrolling within each slide\n        ').tag(config=True)
    reveal_number = Unicode('', help="\n        slide number format (e.g. 'c/t'). Choose from:\n        'c': current, 't': total, 'h': horizontal, 'v': vertical\n        ").tag(config=True)
    reveal_width = Unicode('', help='\n        width used to determine the aspect ratio of your presentation.\n        Use the horizontal pixels available on your intended presentation\n        equipment.\n        ').tag(config=True)
    reveal_height = Unicode('', help='\n        height used to determine the aspect ratio of your presentation.\n        Use the horizontal pixels available on your intended presentation\n        equipment.\n        ').tag(config=True)
    font_awesome_url = Unicode('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css', help='\n        URL to load font awesome from.\n\n        Defaults to loading from cdnjs.\n        ').tag(config=True)

    def _init_resources(self, resources):
        resources = super()._init_resources(resources)
        if 'reveal' not in resources:
            resources['reveal'] = {}
        resources['reveal']['url_prefix'] = self.reveal_url_prefix
        resources['reveal']['theme'] = self.reveal_theme
        resources['reveal']['transition'] = self.reveal_transition
        resources['reveal']['scroll'] = self.reveal_scroll
        resources['reveal']['number'] = self.reveal_number
        resources['reveal']['height'] = self.reveal_height
        resources['reveal']['width'] = self.reveal_width
        return resources