import re
from sphinx.locale import _
from sphinx.ext.napoleon.docstring import NumpyDocstring
def _parse_parameters_section(self, section):
    labels = {'args': _('Parameters'), 'arguments': _('Parameters'), 'parameters': _('Parameters')}
    label = labels.get(section.lower(), section)
    fields = self._consume_fields()
    if self._config.napoleon_use_param:
        return self._format_docutils_params(fields)
    return self._format_fields(label, fields)