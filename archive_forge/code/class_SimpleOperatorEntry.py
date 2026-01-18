from .abstract_impl import AbstractImplHolder
class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self.abstract_impl: AbstractImplHolder = AbstractImplHolder(qualname)