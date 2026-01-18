from typing import TYPE_CHECKING, Any, Dict
from docutils.transforms.references import DanglingReferences
from sphinx.transforms import SphinxTransform
class SphinxDanglingReferences(DanglingReferences):
    """DanglingReferences transform which does not output info messages."""

    def apply(self, **kwargs: Any) -> None:
        try:
            reporter = self.document.reporter
            report_level = reporter.report_level
            reporter.report_level = max(reporter.WARNING_LEVEL, reporter.report_level)
            super().apply()
        finally:
            reporter.report_level = report_level