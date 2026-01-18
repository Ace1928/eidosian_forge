from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
class PtolemyVarietyList(list):

    def retrieve_decomposition(self, *args, **kwargs):
        return MethodMappingList([p.retrieve_decomposition(*args, **kwargs) for p in self])

    def compute_decomposition(self, *args, **kwargs):
        return MethodMappingList([p.compute_decomposition(*args, **kwargs) for p in self])

    def retrieve_solutions(self, *args, **kwargs):
        return MethodMappingList([p.retrieve_solutions(*args, **kwargs) for p in self])

    def compute_solutions(self, *args, **kwargs):
        return MethodMappingList([p.compute_solutions(*args, **kwargs) for p in self])