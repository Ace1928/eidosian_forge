from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
class AttrMapValue:
    """Simple multi-value holder for attribute maps"""

    def __init__(self, validate=None, desc=None, initial=None, advancedUsage=0, **kw):
        self.validate = validate or isAnything
        self.desc = desc
        self._initial = initial
        self._advancedUsage = advancedUsage
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name == 'initial':
            if isinstance(self._initial, CallableValue):
                return self._initial()
            return self._initial
        elif name == 'hidden':
            return 0
        raise AttributeError(name)

    def __repr__(self):
        return 'AttrMapValue(%s)' % ', '.join(['%s=%r' % i for i in self.__dict__.items()])