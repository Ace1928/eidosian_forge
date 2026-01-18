from __future__ import annotations
import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional
import torch
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
from torch.utils import cpp_backtrace
class ExportDiagnosticEngine:
    """PyTorch ONNX Export diagnostic engine.

    The only purpose of creating this class instead of using `DiagnosticContext` directly
    is to provide a background context for `diagnose` calls inside exporter.

    By design, one `torch.onnx.export` call should initialize one diagnostic context.
    All `diagnose` calls inside exporter should be made in the context of that export.
    However, since diagnostic context is currently being accessed via a global variable,
    there is no guarantee that the context is properly initialized. Therefore, we need
    to provide a default background context to fallback to, otherwise any invocation of
    exporter internals, e.g. unit tests, will fail due to missing diagnostic context.
    This can be removed once the pipeline for context to flow through the exporter is
    established.
    """
    contexts: List[infra.DiagnosticContext]
    _background_context: infra.DiagnosticContext

    def __init__(self) -> None:
        self.contexts = []
        self._background_context = infra.DiagnosticContext(name='torch.onnx', version=torch.__version__)

    @property
    def background_context(self) -> infra.DiagnosticContext:
        return self._background_context

    def create_diagnostic_context(self, name: str, version: str, options: Optional[infra.DiagnosticOptions]=None) -> infra.DiagnosticContext:
        """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
        if options is None:
            options = infra.DiagnosticOptions()
        context: infra.DiagnosticContext[infra.Diagnostic] = infra.DiagnosticContext(name, version, options)
        self.contexts.append(context)
        return context

    def clear(self):
        """Clears all diagnostic contexts."""
        self.contexts.clear()
        self._background_context.diagnostics.clear()

    def to_json(self) -> str:
        return formatter.sarif_to_json(self.sarif_log())

    def dump(self, file_path: str, compress: bool=False) -> None:
        """Dumps the SARIF log to a file."""
        if compress:
            with gzip.open(file_path, 'wt') as f:
                f.write(self.to_json())
        else:
            with open(file_path, 'w') as f:
                f.write(self.to_json())

    def sarif_log(self):
        log = sarif.SarifLog(version=sarif_version.SARIF_VERSION, schema_uri=sarif_version.SARIF_SCHEMA_LINK, runs=[context.sarif() for context in self.contexts])
        log.runs.append(self._background_context.sarif())
        return log