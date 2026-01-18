from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
@dataclasses.dataclass
class DiagnosticContext(Generic[_Diagnostic]):
    name: str
    version: str
    options: infra.DiagnosticOptions = dataclasses.field(default_factory=infra.DiagnosticOptions)
    diagnostics: List[_Diagnostic] = dataclasses.field(init=False, default_factory=list)
    _inflight_diagnostics: List[_Diagnostic] = dataclasses.field(init=False, default_factory=list)
    _previous_log_level: int = dataclasses.field(init=False, default=logging.WARNING)
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    _bound_diagnostic_type: Type = dataclasses.field(init=False, default=Diagnostic)

    def __enter__(self):
        self._previous_log_level = self.logger.level
        self.logger.setLevel(self.options.verbosity_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self._previous_log_level)
        return None

    def sarif(self) -> sarif.Run:
        """Returns the SARIF Run object."""
        unique_rules = {diagnostic.rule for diagnostic in self.diagnostics}
        return sarif.Run(sarif.Tool(driver=sarif.ToolComponent(name=self.name, version=self.version, rules=[rule.sarif() for rule in unique_rules])), results=[diagnostic.sarif() for diagnostic in self.diagnostics])

    def sarif_log(self) -> sarif.SarifLog:
        """Returns the SARIF Log object."""
        return sarif.SarifLog(version=sarif_version.SARIF_VERSION, schema_uri=sarif_version.SARIF_SCHEMA_LINK, runs=[self.sarif()])

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

    def log(self, diagnostic: _Diagnostic) -> None:
        """Logs a diagnostic.

        This method should be used only after all the necessary information for the diagnostic
        has been collected.

        Args:
            diagnostic: The diagnostic to add.
        """
        if not isinstance(diagnostic, self._bound_diagnostic_type):
            raise TypeError(f'Expected diagnostic of type {self._bound_diagnostic_type}, got {type(diagnostic)}')
        if self.options.warnings_as_errors and diagnostic.level == infra.Level.WARNING:
            diagnostic.level = infra.Level.ERROR
        self.diagnostics.append(diagnostic)

    def log_and_raise_if_error(self, diagnostic: _Diagnostic) -> None:
        """Logs a diagnostic and raises an exception if it is an error.

        Use this method for logging non inflight diagnostics where diagnostic level is not known or
        lower than ERROR. If it is always expected raise, use `log` and explicit
        `raise` instead. Otherwise there is no way to convey the message that it always
        raises to Python intellisense and type checking tools.

        This method should be used only after all the necessary information for the diagnostic
        has been collected.

        Args:
            diagnostic: The diagnostic to add.
        """
        self.log(diagnostic)
        if diagnostic.level == infra.Level.ERROR:
            if diagnostic.source_exception is not None:
                raise diagnostic.source_exception
            raise RuntimeErrorWithDiagnostic(diagnostic)

    @contextlib.contextmanager
    def add_inflight_diagnostic(self, diagnostic: _Diagnostic) -> Generator[_Diagnostic, None, None]:
        """Adds a diagnostic to the context.

        Use this method to add diagnostics that are not created by the context.
        Args:
            diagnostic: The diagnostic to add.
        """
        self._inflight_diagnostics.append(diagnostic)
        try:
            yield diagnostic
        finally:
            self._inflight_diagnostics.pop()

    def push_inflight_diagnostic(self, diagnostic: _Diagnostic) -> None:
        """Pushes a diagnostic to the inflight diagnostics stack.

        Args:
            diagnostic: The diagnostic to push.

        Raises:
            ValueError: If the rule is not supported by the tool.
        """
        self._inflight_diagnostics.append(diagnostic)

    def pop_inflight_diagnostic(self) -> _Diagnostic:
        """Pops the last diagnostic from the inflight diagnostics stack.

        Returns:
            The popped diagnostic.
        """
        return self._inflight_diagnostics.pop()

    def inflight_diagnostic(self, rule: Optional[infra.Rule]=None) -> _Diagnostic:
        if rule is None:
            if len(self._inflight_diagnostics) <= 0:
                raise AssertionError('No inflight diagnostics')
            return self._inflight_diagnostics[-1]
        else:
            for diagnostic in reversed(self._inflight_diagnostics):
                if diagnostic.rule == rule:
                    return diagnostic
            raise AssertionError(f'No inflight diagnostic for rule {rule.name}')