import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def _get_doc_link(self):
    """Generates a link to the API documentation for a given estimator.

        This method generates the link to the estimator's documentation page
        by using the template defined by the attribute `_doc_link_template`.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
    if self.__class__.__module__.split('.')[0] != self._doc_link_module:
        return ''
    if self._doc_link_url_param_generator is None:
        estimator_name = self.__class__.__name__
        estimator_module = '.'.join(itertools.takewhile(lambda part: not part.startswith('_'), self.__class__.__module__.split('.')))
        return self._doc_link_template.format(estimator_module=estimator_module, estimator_name=estimator_name)
    return self._doc_link_template.format(**self._doc_link_url_param_generator(self))