from ..overrides import override
from ..module import get_introspection_module
class OverridesStruct(GIMarshallingTests.OverridesStruct):

    def __new__(cls, long_):
        return GIMarshallingTests.OverridesStruct.__new__(cls)

    def __init__(self, long_):
        GIMarshallingTests.OverridesStruct.__init__(self)
        self.long_ = long_

    def method(self):
        return GIMarshallingTests.OverridesStruct.method(self) / 7