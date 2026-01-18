from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_property
class ComponentPrinter(cp.CustomPrinterBase):
    """Prints the KubeRun Component custom human-readable format."""

    def Transform(self, record):
        """Transform a service into the output structure of marker classes."""
        sections = [self._Header(record), self._SpecSection(record)] + self._ConfigSections(record)
        return cp.Lines(_Spaced(sections))

    def _Header(self, record):
        con = console_attr.GetConsoleAttr()
        return con.Emphasize('Component {}'.format(record['metadata']['name']))

    def _SpecSection(self, record):
        spec = record.get('spec', {})
        return cp.Section([cp.Labeled([('Type', spec.get('type', '')), ('DevKit', spec.get('devkit', '')), ('DevKit Version', spec.get('devkit-version', ''))])])

    def _ConfigSections(self, record):
        config = record.get('spec', {}).get('config', {})
        sections = []
        for section_name, data in sorted(config.items()):
            title = _ConfigTitle(section_name)
            section = cp.Section([cp.Labeled([(title, _ConfigSectionData(data))])])
            sections.append(section)
        return sections