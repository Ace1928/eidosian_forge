from jedi.inference import compiled
from jedi.inference.base_value import ValueSet
from jedi.inference.filters import ParserTreeFilter, MergedFilter
from jedi.inference.names import TreeNameDefinition
from jedi.inference.compiled import mixed
from jedi.inference.compiled.access import create_access_path
from jedi.inference.context import ModuleContext
class MixedTreeName(TreeNameDefinition):

    def infer(self):
        """
        In IPython notebook it is typical that some parts of the code that is
        provided was already executed. In that case if something is not properly
        inferred, it should still infer from the variables it already knows.
        """
        inferred = super().infer()
        if not inferred:
            for compiled_value in self.parent_context.mixed_values:
                for f in compiled_value.get_filters():
                    values = ValueSet.from_sets((n.infer() for n in f.get(self.string_name)))
                    if values:
                        return values
        return inferred