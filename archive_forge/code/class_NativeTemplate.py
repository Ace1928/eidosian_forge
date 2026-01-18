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
class NativeTemplate(Template):
    environment_class = NativeEnvironment

    def render(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Render the template to produce a native Python type. If the
        result is a single node, its value is returned. Otherwise, the
        nodes are concatenated as strings. If the result can be parsed
        with :func:`ast.literal_eval`, the parsed value is returned.
        Otherwise, the string is returned.
        """
        ctx = self.new_context(dict(*args, **kwargs))
        try:
            return self.environment_class.concat(self.root_render_func(ctx))
        except Exception:
            return self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if not self.environment.is_async:
            raise RuntimeError('The environment was not created with async mode enabled.')
        ctx = self.new_context(dict(*args, **kwargs))
        try:
            return self.environment_class.concat([n async for n in self.root_render_func(ctx)])
        except Exception:
            return self.environment.handle_exception()