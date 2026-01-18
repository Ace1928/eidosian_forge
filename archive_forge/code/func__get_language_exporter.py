import sys
from traitlets import Dict, default
from .base import get_exporter
from .templateexporter import TemplateExporter
def _get_language_exporter(self, lang_name):
    """Find an exporter for the language name from notebook metadata.

        Uses the nbconvert.exporters.script group of entry points.
        Returns None if no exporter is found.
        """
    if lang_name not in self._lang_exporters:
        try:
            exporters = entry_points(group='nbconvert.exporters.script')
            exporter = [e for e in exporters if e.name == lang_name][0].load()
        except (KeyError, IndexError):
            self._lang_exporters[lang_name] = None
        else:
            self._lang_exporters[lang_name] = exporter(config=self.config, parent=self)
    return self._lang_exporters[lang_name]