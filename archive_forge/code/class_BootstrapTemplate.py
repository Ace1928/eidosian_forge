from __future__ import annotations
import pathlib
from typing import ClassVar, Dict, List
import param
from ...theme import Design
from ...theme.bootstrap import Bootstrap
from ..base import BasicTemplate, TemplateActions
class BootstrapTemplate(BasicTemplate):
    """
    BootstrapTemplate
    """
    design = param.ClassSelector(class_=Design, default=Bootstrap, is_instance=False, instantiate=False, doc='\n        A Design applies a specific design system to a template.')
    sidebar_width = param.Integer(default=350, doc='\n        The width of the sidebar in pixels. Default is 350.')
    _actions = param.ClassSelector(default=BootstrapTemplateActions(), class_=TemplateActions)
    _css = [_ROOT / 'bootstrap.css']
    _template = _ROOT / 'bootstrap.html'

    def _update_vars(self, *args) -> None:
        super()._update_vars(*args)
        design = self.design(theme=self.theme)
        self._render_variables['html_attrs'] = f'data-bs-theme="{design.theme._bs_theme}"'