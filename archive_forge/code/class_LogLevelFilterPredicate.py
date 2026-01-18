from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver
@implementer(ILogFilterPredicate)
class LogLevelFilterPredicate:
    """
    L{ILogFilterPredicate} that filters out events with a log level lower than
    the log level for the event's namespace.

    Events that not not have a log level or namespace are also dropped.
    """

    def __init__(self, defaultLogLevel: NamedConstant=LogLevel.info) -> None:
        """
        @param defaultLogLevel: The default minimum log level.
        """
        self._logLevelsByNamespace: Dict[str, NamedConstant] = {}
        self.defaultLogLevel = defaultLogLevel
        self.clearLogLevels()

    def logLevelForNamespace(self, namespace: str) -> NamedConstant:
        """
        Determine an appropriate log level for the given namespace.

        This respects dots in namespaces; for example, if you have previously
        invoked C{setLogLevelForNamespace("mypackage", LogLevel.debug)}, then
        C{logLevelForNamespace("mypackage.subpackage")} will return
        C{LogLevel.debug}.

        @param namespace: A logging namespace.  Use C{""} for the default
            namespace.

        @return: The log level for the specified namespace.
        """
        if not namespace:
            return self._logLevelsByNamespace['']
        if namespace in self._logLevelsByNamespace:
            return self._logLevelsByNamespace[namespace]
        segments = namespace.split('.')
        index = len(segments) - 1
        while index > 0:
            namespace = '.'.join(segments[:index])
            if namespace in self._logLevelsByNamespace:
                return self._logLevelsByNamespace[namespace]
            index -= 1
        return self._logLevelsByNamespace['']

    def setLogLevelForNamespace(self, namespace: str, level: NamedConstant) -> None:
        """
        Sets the log level for a logging namespace.

        @param namespace: A logging namespace.
        @param level: The log level for the given namespace.
        """
        if level not in LogLevel.iterconstants():
            raise InvalidLogLevelError(level)
        if namespace:
            self._logLevelsByNamespace[namespace] = level
        else:
            self._logLevelsByNamespace[''] = level

    def clearLogLevels(self) -> None:
        """
        Clears all log levels to the default.
        """
        self._logLevelsByNamespace.clear()
        self._logLevelsByNamespace[''] = self.defaultLogLevel

    def __call__(self, event: LogEvent) -> NamedConstant:
        eventLevel = event.get('log_level', None)
        if eventLevel is None:
            return PredicateResult.no
        namespace = event.get('log_namespace', '')
        if not namespace:
            return PredicateResult.no
        namespaceLevel = self.logLevelForNamespace(namespace)
        if eventLevel < namespaceLevel:
            return PredicateResult.no
        return PredicateResult.maybe