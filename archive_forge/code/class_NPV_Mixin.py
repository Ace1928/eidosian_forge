import enum
from pyomo.common.dependencies import attempt_import
from pyomo.common.numeric_types import native_types
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import OperatorAssociativity
class NPV_Mixin(object):
    __slots__ = ()

    def is_potentially_variable(self):
        return False

    def create_node_with_local_data(self, args, classtype=None):
        assert classtype is None
        try:
            npv_args = all((type(arg) in native_types or not arg.is_potentially_variable() for arg in args))
        except AttributeError:
            npv_args = False
        if npv_args:
            return super().create_node_with_local_data(args, None)
        else:
            return super().create_node_with_local_data(args, self.potentially_variable_base_class())

    def potentially_variable_base_class(self):
        cls = list(self.__class__.__bases__)
        cls.remove(NPV_Mixin)
        assert len(cls) == 1
        return cls[0]