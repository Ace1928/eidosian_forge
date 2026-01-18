from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
import six
class PropertyDiagnostic(diagnostic_base.Diagnostic):
    """Diagnoses issues that may be caused by properties."""

    def __init__(self, ignore_hidden_property_allowlist):
        intro = 'Property diagnostic detects issues that may be caused by properties.'
        super(PropertyDiagnostic, self).__init__(intro=intro, title='Property diagnostic', checklist=[HiddenPropertiesChecker(ignore_hidden_property_allowlist)])