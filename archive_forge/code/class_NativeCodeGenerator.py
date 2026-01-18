import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template
class NativeCodeGenerator(CodeGenerator):
    """A code generator which renders Python types by not adding
    ``str()`` around output nodes.
    """

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        return value

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        return repr(''.join([str(v) for v in group]))

    def _output_child_to_const(self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo) -> t.Any:
        const = node.as_const(frame.eval_ctx)
        if not has_safe_repr(const):
            raise nodes.Impossible()
        if isinstance(node, nodes.TemplateData):
            return const
        return finalize.const(const)

    def _output_child_pre(self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo) -> None:
        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo) -> None:
        if finalize.src is not None:
            self.write(')')