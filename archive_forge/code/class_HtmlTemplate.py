import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
class HtmlTemplate:
    """Contain html templates for InferenceData repr."""
    html_template = '\n            <div>\n              <div class=\'xr-header\'>\n                <div class="xr-obj-type">arviz.InferenceData</div>\n              </div>\n              <ul class="xr-sections group-sections">\n              {}\n              </ul>\n            </div>\n            '
    element_template = '\n            <li class = "xr-section-item">\n                  <input id="idata_{group_id}" class="xr-section-summary-in" type="checkbox">\n                  <label for="idata_{group_id}" class = "xr-section-summary">{group}</label>\n                  <div class="xr-section-inline-details"></div>\n                  <div class="xr-section-details">\n                      <ul id="xr-dataset-coord-list" class="xr-var-list">\n                          <div style="padding-left:2rem;">{xr_data}<br></div>\n                      </ul>\n                  </div>\n            </li>\n            '
    _, css_style = _load_static_files()
    specific_style = '.xr-wrap{width:700px!important;}'
    css_template = f'<style> {css_style}{specific_style} </style>'