from ipywidgets import DOMWidget
from traitlets import Unicode, List, Bool
from ._version import __frontend_version__
Default tooltip widget for marks.

    Attributes
    ----------
    fields: list (default: [])
        list of names of fields to be displayed in the tooltip
        All the attributes  of the mark are accessible in the tooltip
    formats: list (default: [])
        list of formats to be applied to each of the fields.
        if no format is specified for a field, the value is displayed as it is
    labels: list (default: [])
        list of labels to be displayed in the table instead of the fields. If
        the length of labels is less than the length of fields, then the field
        names are displayed for those fields for which label is missing.
    show_labels: bool (default: True)
        Boolean attribute to enable and disable display of the
        label /field name
        as the first column along with the value
    