import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def _write_estimator_html(out, estimator, estimator_label, estimator_label_details, is_fitted_css_class, is_fitted_icon='', first_call=False):
    """Write estimator to html in serial, parallel, or by itself (single).

    For multiple estimators, this function is called recursively.

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    estimator : estimator object
        The estimator to visualize.
    estimator_label : str
        The label for the estimator. It corresponds either to the estimator class name
        for simple estimator or in the case of `Pipeline` and `ColumnTransformer`, it
        corresponds to the name of the step.
    estimator_label_details : str
        The details to show as content in the dropdown part of the toggleable label.
        It can contain information as non-default parameters or column information for
        `ColumnTransformer`.
    is_fitted_css_class : {"", "fitted"}
        The CSS class to indicate whether or not the estimator is fitted or not. The
        empty string means that the estimator is not fitted and "fitted" means that the
        estimator is fitted.
    is_fitted_icon : str, default=""
        The HTML representation to show the fitted information in the diagram. An empty
        string means that no information is shown. If the estimator to be shown is not
        the first estimator (i.e. `first_call=False`), `is_fitted_icon` is always an
        empty string.
    first_call : bool, default=False
        Whether this is the first time this function is called.
    """
    if first_call:
        est_block = _get_visual_block(estimator)
    else:
        is_fitted_icon = ''
        with config_context(print_changed_only=True):
            est_block = _get_visual_block(estimator)
    if hasattr(estimator, '_get_doc_link'):
        doc_link = estimator._get_doc_link()
    else:
        doc_link = ''
    if est_block.kind in ('serial', 'parallel'):
        dashed_wrapped = first_call or est_block.dash_wrapped
        dash_cls = ' sk-dashed-wrapped' if dashed_wrapped else ''
        out.write(f'<div class="sk-item{dash_cls}">')
        if estimator_label:
            _write_label_html(out, estimator_label, estimator_label_details, doc_link=doc_link, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)
        kind = est_block.kind
        out.write(f'<div class="sk-{kind}">')
        est_infos = zip(est_block.estimators, est_block.names, est_block.name_details)
        for est, name, name_details in est_infos:
            if kind == 'serial':
                _write_estimator_html(out, est, name, name_details, is_fitted_css_class=is_fitted_css_class)
            else:
                out.write('<div class="sk-parallel-item">')
                serial_block = _VisualBlock('serial', [est], dash_wrapped=False)
                _write_estimator_html(out, serial_block, name, name_details, is_fitted_css_class=is_fitted_css_class)
                out.write('</div>')
        out.write('</div></div>')
    elif est_block.kind == 'single':
        _write_label_html(out, est_block.names, est_block.name_details, outer_class='sk-item', inner_class='sk-estimator', checked=first_call, doc_link=doc_link, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)