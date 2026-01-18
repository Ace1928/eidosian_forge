from jedi.inference import compiled
from jedi.inference.base_value import ValueSet
from jedi.inference.filters import ParserTreeFilter, MergedFilter
from jedi.inference.names import TreeNameDefinition
from jedi.inference.compiled import mixed
from jedi.inference.compiled.access import create_access_path
from jedi.inference.context import ModuleContext
class MixedParserTreeFilter(ParserTreeFilter):
    name_class = MixedTreeName