import pathlib
import param
from ...config import config as pn_config
from ..vanilla import VanillaTemplate
@param.depends('reveal_config', 'reveal_theme', 'show_header', watch=True)
def _update_render_vars(self):
    self._resources['css'] = {'font': FONT_CSS, 'reveal': REVEAL_CSS, 'reveal-theme': REVEAL_THEME_CSS[f'reveal-{self.reveal_theme}']}
    self._render_variables['show_header'] = self.show_header
    self._render_variables['reveal_config'] = self.reveal_config