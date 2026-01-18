from typing import Dict, Optional
from .config import LogLevel
from .logger_decorator import enable_logging
class ClassLogger:
    """
    Ensure all subclasses of the class being inherited are logged, too.

    Notes
    -----
    This mixin must go first in class bases declaration to have the desired effect.
    """
    _modin_logging_layer = 'PANDAS-API'

    @classmethod
    def __init_subclass__(cls, modin_layer: Optional[str]=None, class_name: Optional[str]=None, log_level: LogLevel=LogLevel.INFO, **kwargs: Dict) -> None:
        """
        Apply logging decorator to all children of ``ClassLogger``.

        Parameters
        ----------
        modin_layer : str, default: "PANDAS-API"
            Specified by the logger (e.g. PANDAS-API).
        class_name : str, optional
            The name of the class the decorator is being applied to.
            Composed from the decorated class name if not specified.
        log_level : LogLevel, default: LogLevel.INFO
            The log level (LogLevel.INFO, LogLevel.DEBUG, LogLevel.WARNING, etc.).
        **kwargs : dict
        """
        modin_layer = modin_layer or cls._modin_logging_layer
        super().__init_subclass__(**kwargs)
        enable_logging(modin_layer, class_name, log_level)(cls)
        cls._modin_logging_layer = modin_layer