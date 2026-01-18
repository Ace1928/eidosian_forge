import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _resolveFunctionGroup(self, functionDict, interactiveFunction):
    """
        Returns parent parameter that holds function children. May be ``None`` if
        no top parent is provided and nesting is disabled.
        """
    funcGroup = self.parent
    if self.nest:
        funcGroup = Parameter.create(**functionDict)
        if self.parent:
            funcGroup = self.parent.addChild(funcGroup, existOk=self.existOk)
        funcGroup.sigActivated.connect(interactiveFunction.runFromAction)
    return funcGroup