import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def _write_label_html(out, name, name_details, outer_class='sk-label-container', inner_class='sk-label', checked=False, doc_link='', is_fitted_css_class='', is_fitted_icon=''):
    """Write labeled html with or without a dropdown with named details.

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    name : str
        The label for the estimator. It corresponds either to the estimator class name
        for a simple estimator or in the case of a `Pipeline` and `ColumnTransformer`,
        it corresponds to the name of the step.
    name_details : str
        The details to show as content in the dropdown part of the toggleable label. It
        can contain information such as non-default parameters or column information for
        `ColumnTransformer`.
    outer_class : {"sk-label-container", "sk-item"}, default="sk-label-container"
        The CSS class for the outer container.
    inner_class : {"sk-label", "sk-estimator"}, default="sk-label"
        The CSS class for the inner container.
    checked : bool, default=False
        Whether the dropdown is folded or not. With a single estimator, we intend to
        unfold the content.
    doc_link : str, default=""
        The link to the documentation for the estimator. If an empty string, no link is
        added to the diagram. This can be generated for an estimator if it uses the
        `_HTMLDocumentationLinkMixin`.
    is_fitted_css_class : {"", "fitted"}
        The CSS class to indicate whether or not the estimator is fitted. The
        empty string means that the estimator is not fitted and "fitted" means that the
        estimator is fitted.
    is_fitted_icon : str, default=""
        The HTML representation to show the fitted information in the diagram. An empty
        string means that no information is shown.
    """
    padding_label = '&nbsp;' if is_fitted_icon else ''
    out.write(f'<div class="{outer_class}"><div class="{inner_class} {is_fitted_css_class} sk-toggleable">')
    name = html.escape(name)
    if name_details is not None:
        name_details = html.escape(str(name_details))
        label_class = f'sk-toggleable__label {is_fitted_css_class} sk-toggleable__label-arrow'
        checked_str = 'checked' if checked else ''
        est_id = _ESTIMATOR_ID_COUNTER.get_id()
        if doc_link:
            doc_label = '<span>Online documentation</span>'
            if name is not None:
                doc_label = f'<span>Documentation for {name}</span>'
            doc_link = f'<a class="sk-estimator-doc-link {is_fitted_css_class}" rel="noreferrer" target="_blank" href="{doc_link}">?{doc_label}</a>'
            padding_label += '&nbsp;'
        fmt_str = f'<input class="sk-toggleable__control sk-hidden--visually" id="{est_id}" type="checkbox" {checked_str}><label for="{est_id}" class="{label_class} {is_fitted_css_class}">{padding_label}{name}{doc_link}{is_fitted_icon}</label><div class="sk-toggleable__content {is_fitted_css_class}"><pre>{name_details}</pre></div> '
        out.write(fmt_str)
    else:
        out.write(f'<label>{name}</label>')
    out.write('</div></div>')