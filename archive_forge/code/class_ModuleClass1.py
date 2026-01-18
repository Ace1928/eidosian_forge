from tensorflow.tools.common import test_module2
class ModuleClass1(object):

    def __init__(self):
        self._m2 = test_module2.ModuleClass2()

    def __model_class1_method__(self):
        pass