from os import path
from typing import Any, Dict, Iterator, Optional, Set, Type, Union
from docutils import nodes
from docutils.io import StringOutput
from docutils.nodes import Node
from docutils.writers.docutils_xml import XMLTranslator
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import ensuredir, os_path
from sphinx.writers.xml import PseudoXMLWriter, XMLWriter
class PseudoXMLBuilder(XMLBuilder):
    """
    Builds pseudo-XML for display purposes.
    """
    name = 'pseudoxml'
    format = 'pseudoxml'
    epilog = __('The pseudo-XML files are in %(outdir)s.')
    out_suffix = '.pseudoxml'
    _writer_class = PseudoXMLWriter