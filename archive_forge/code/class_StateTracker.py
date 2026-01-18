from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
class StateTracker:
    """Keeps a stack of transforms and state
    properties.  It can contain any properties you
    want, but the keys 'transform' and 'ctm' have
    special meanings.  The getCTM()
    method returns the current transformation
    matrix at any point, without needing to
    invert matrixes when you pop."""

    def __init__(self, defaults=None, defaultObj=None):
        self._deltas = []
        self._combined = []
        if defaults is None:
            defaults = STATE_DEFAULTS.copy()
        if defaultObj:
            for k in STATE_DEFAULTS.keys():
                a = 'initial' + k[:1].upper() + k[1:]
                if hasattr(defaultObj, a):
                    defaults[k] = getattr(defaultObj, a)
        if 'transform' in defaults:
            defaults['ctm'] = defaults['transform']
        self._combined.append(defaults)

    def _applyDefaultObj(self, d):
        return d

    def push(self, delta):
        """Take a new state dictionary of changes and push it onto
        the stack.  After doing this, the combined state is accessible
        through getState()"""
        newstate = self._combined[-1].copy()
        for key, value in delta.items():
            if key == 'transform':
                newstate['transform'] = delta['transform']
                newstate['ctm'] = mmult(self._combined[-1]['ctm'], delta['transform'])
            else:
                newstate[key] = value
        self._combined.append(newstate)
        self._deltas.append(delta)

    def pop(self):
        """steps back one, and returns a state dictionary with the
        deltas to reverse out of wherever you are.  Depending
        on your back end, you may not need the return value,
        since you can get the complete state afterwards with getState()"""
        del self._combined[-1]
        newState = self._combined[-1]
        lastDelta = self._deltas[-1]
        del self._deltas[-1]
        reverseDelta = {}
        for key, curValue in lastDelta.items():
            prevValue = newState[key]
            if prevValue != curValue:
                if key == 'transform':
                    reverseDelta[key] = inverse(lastDelta['transform'])
                else:
                    reverseDelta[key] = prevValue
        return reverseDelta

    def getState(self):
        """returns the complete graphics state at this point"""
        return self._combined[-1]

    def getCTM(self):
        """returns the current transformation matrix at this point"""
        return self._combined[-1]['ctm']

    def __getitem__(self, key):
        """returns the complete graphics state value of key at this point"""
        return self._combined[-1][key]

    def __setitem__(self, key, value):
        """sets the complete graphics state value of key to value"""
        self._combined[-1][key] = value