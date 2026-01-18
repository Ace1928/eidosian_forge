from torch import nn
class OrderedDictWrapper:
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr):
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)

    def items(self):
        return self.cpp_dict.items()

    def keys(self):
        return self.cpp_dict.keys()

    def values(self):
        return self.cpp_dict.values()

    def __iter__(self):
        return self.cpp_dict.__iter__()

    def __len__(self):
        return self.cpp_dict.__len__()

    def __contains__(self, key):
        return self.cpp_dict.__contains__(key)

    def __getitem__(self, key):
        return self.cpp_dict.__getitem__(key)