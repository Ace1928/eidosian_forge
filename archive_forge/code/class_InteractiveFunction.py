import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
class InteractiveFunction:
    """
    ``interact`` can be used with regular functions. However, when they are connected to
    changed or changing signals, there is no way to access these connections later to
    i.e. disconnect them temporarily. This utility class wraps a normal function but
    can provide an external scope for accessing the hooked up parameter signals.
    """
    __name__: str
    __qualname__: str

    def __init__(self, function, *, closures=None, **extra):
        """
        Wraps a callable function in a way that forwards Parameter arguments as keywords

        Parameters
        ----------
        function: callable
            Function to wrap
        closures: dict[str, callable]
            Arguments that shouldn't be constant, but can't be represented as a parameter.
            See the rst docs for more information.
        extra: dict
            extra keyword arguments to pass to ``function`` when this wrapper is called
        """
        super().__init__()
        self.parameters = {}
        self.extra = extra
        self.function = function
        if closures is None:
            closures = {}
        self.closures = closures
        self._disconnected = False
        self.parametersNeedRunKwargs = False
        self.parameterCache = {}
        functools.update_wrapper(self, function, updated=())

    def __call__(self, **kwargs):
        """
        Calls ``self.function``. Extra, closures, and parameter keywords as defined on
        init and through :func:`InteractiveFunction.setParams` are forwarded during the
        call.
        """
        if self.parametersNeedRunKwargs:
            self._updateParametersFromRunKwargs(**kwargs)
        runKwargs = self.extra.copy()
        runKwargs.update(self.parameterCache)
        for kk, vv in self.closures.items():
            runKwargs[kk] = vv()
        runKwargs.update(**kwargs)
        return self.function(**runKwargs)

    def updateCachedParameterValues(self, param, value):
        """
        This function is connected to ``sigChanged`` of every parameter associated with
        it. This way, those parameters don't have to be queried for their value every
        time InteractiveFunction is __call__'ed
        """
        self.parameterCache[param.name()] = value

    def _updateParametersFromRunKwargs(self, **kwargs):
        """
        Updates attached params from __call__ without causing additional function runs
        """
        wasDisconnected = self.disconnect()
        try:
            for kwarg in set(kwargs).intersection(self.parameters):
                self.parameters[kwarg].setValue(kwargs[kwarg])
        finally:
            if not wasDisconnected:
                self.reconnect()
        for extraKey in set(kwargs) & set(self.extra):
            self.extra[extraKey] = kwargs[extraKey]

    def _disconnectParameter(self, param):
        param.sigValueChanged.disconnect(self.updateCachedParameterValues)
        for signal in (param.sigValueChanging, param.sigValueChanged):
            fn.disconnect(signal, self.runFromChangedOrChanging)

    def hookupParameters(self, params=None, clearOld=True):
        """
        Binds a new set of parameters to this function. If ``clearOld`` is *True* (
        default), previously bound parameters are disconnected.

        Parameters
        ----------
        params: Sequence[Parameter]
            New parameters to listen for updates and optionally propagate keywords
            passed to :meth:`__call__`
        clearOld: bool
            If ``True``, previously hooked up parameters will be removed first
        """
        if clearOld:
            self.removeParameters()
        for param in params:
            self.parameters[param.name()] = param
            param.sigValueChanged.connect(self.updateCachedParameterValues)
            self.parameterCache[param.name()] = param.value() if param.hasValue() else None

    def removeParameters(self, clearCache=True):
        """
        Disconnects from all signals of parameters in ``self.parameters``. Also,
        optionally clears the old cache of param values
        """
        for p in self.parameters.values():
            self._disconnectParameter(p)
        self.parameters.clear()
        if clearCache:
            self.parameterCache.clear()

    def runFromChangedOrChanging(self, param, value):
        if self._disconnected:
            return None
        oldPropagate = self.parametersNeedRunKwargs
        self.parametersNeedRunKwargs = False
        try:
            ret = self(**{param.name(): value})
        finally:
            self.parametersNeedRunKwargs = oldPropagate
        return ret

    def runFromAction(self, **kwargs):
        if self._disconnected:
            return None
        return self(**kwargs)

    def disconnect(self):
        """
        Simulates disconnecting the runnable by turning ``runFrom*`` functions into no-ops
        """
        oldDisconnect = self._disconnected
        self._disconnected = True
        return oldDisconnect

    def setDisconnected(self, disconnected):
        """
        Sets the disconnected state of the runnable, see :meth:`disconnect` and
        :meth:`reconnect` for more information
        """
        oldDisconnect = self._disconnected
        self._disconnected = disconnected
        return oldDisconnect

    def reconnect(self):
        """Simulates reconnecting the runnable by re-enabling ``runFrom*`` functions"""
        oldDisconnect = self._disconnected
        self._disconnected = False
        return oldDisconnect

    def __str__(self):
        return f'{type(self).__name__}(`<{self.function.__name__}>`) at {hex(id(self))}'

    def __repr__(self):
        return str(self) + f' with keys:\nparameters={list(self.parameters)}, extra={list(self.extra)}, closures={list(self.closures)}'