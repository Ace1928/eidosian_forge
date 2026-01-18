from numba import types
from numba.core import config
@register_model(TestStructModelType)
class TestStructModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('x', int32), ('y', int32)]
        super().__init__(dmm, fe_type, members)