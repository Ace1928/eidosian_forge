# Import statements enhanced for clarity and completeness
import sys  # Provides access to system-specific parameters and functions. https://docs.python.org/3/library/sys.html
import os  # Provides a way of using operating system dependent functionality. https://docs.python.org/3/library/os.html
import datetime  # Supplies classes for manipulating dates and times. https://docs.python.org/3/library/datetime.html
import logging  # Defines functions and classes which implement a flexible event logging system. https://docs.python.org/3/library/logging.html
from typing import (  # Defines a standard notation for Python function and variable type annotations. https://docs.python.org/3/library/typing.html
    TextIO,  # A generic version of typing.IO[str]. https://docs.python.org/3/library/typing.html#typing.TextIO
    Optional,  # Optional type. https://docs.python.org/3/library/typing.html#typing.Optional
    Any,  # Special type indicating an unconstrained type. https://docs.python.org/3/library/typing.html#typing.Any
    Callable,  # Callable type; Callable[[int], str] is a function of (int) -> str. https://docs.python.org/3/library/typing.html#typing.Callable
    Type,  # A variable annotated with C may accept a value of type C. In contrast, a variable annotated with Type[C] may accept values that are classes themselves – specifically, it will accept the class object of C. https://docs.python.org/3/library/typing.html#typing.Type
    Tuple,  # Tuple type; Tuple[X, Y] is the type of a tuple of two items with the first item of type X and the second of type Y. https://docs.python.org/3/library/typing.html#typing.Tuple
    Dict,  # A generic version of dict. https://docs.python.org/3/library/typing.html#typing.Dict
    Union,  # Union type; Union[X, Y] means either X or Y. https://docs.python.org/3/library/typing.html#typing.Union
    Literal,  # Special typing form to define literal types. https://docs.python.org/3/library/typing.html#typing.Literal
    Iterable,  # A generic version of collections.abc.Iterable. https://docs.python.org/3/library/typing.html#typing.Iterable
    TypeVar,  # Type variable. https://docs.python.org/3/library/typing.html#typing.TypeVar
    Generic,  # Abstract base class for generic types. https://docs.python.org/3/library/typing.html#typing.Generic
    overload,  # Function decorator for defining overloaded functions. https://docs.python.org/3/library/typing.html#typing.overload
    List,  # A generic version of list. https://docs.python.org/3/library/typing.html#typing.List
    Protocol,  # Base class for protocol classes. https://docs.python.org/3/library/typing.html#typing.Protocol
    TypeAlias,  # Special annotation for explicitly declaring a type alias. https://docs.python.org/3/library/typing.html#typing.TypeAlias
)
import traceback  # Provides a standard interface to extract, format and print stack traces of Python programs. https://docs.python.org/3/library/traceback.html
import json  # Implements a subset of the JSON (JavaScript Object Notation) data interchange format. https://docs.python.org/3/library/json.html
from pathlib import (
    Path,
)  # Object-oriented filesystem paths. https://docs.python.org/3/library/pathlib.html
from collections.abc import (
    Iterable as IterableABC,
)  # Abstract base classes for containers. https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
from functools import (
    partial,
    singledispatch,
)  # Higher-order functions and operations on callable objects. https://docs.python.org/3/library/functools.html
from dataclasses import (
    dataclass,
    field,
    asdict,
)  # Provides a decorator and functions for automatically adding generated special methods to user-defined classes. https://docs.python.org/3/library/dataclasses.html
from enum import (
    Enum,
    auto,
    unique,
)  # Support for enumerations. https://docs.python.org/3/library/enum.html
from typing_extensions import (  # Provides additional facilities to the typing module. https://pypi.org/project/typing-extensions/
    Literal,  # Special typing form to define literal types. https://docs.python.org/3/library/typing.html#typing.Literal
    Self,  # Special type to represent the current class type. https://peps.python.org/pep-0673/
    TypeAlias,  # Special annotation for explicitly declaring a type alias. https://docs.python.org/3/library/typing.html#typing.TypeAlias
    Unpack,  # Special typing construct to unpack a variadic type. https://peps.python.org/pep-0646/
    ParamSpec,  # Parameter specification variable. https://peps.python.org/pep-0612/
    TypeVarTuple,  # Type variable tuple. https://peps.python.org/pep-0646/
    Protocol,  # Base class for protocol classes. https://docs.python.org/3/library/typing.html#typing.Protocol
)
from logging.handlers import QueueHandler, QueueListener
import queue
import asyncio
import multiprocessing
import concurrent.futures
from enum import Enum
from typing import TypeAlias, Optional, Tuple, Dict, Union, Literal
from logging import LogRecord


from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Literal, TypeAlias
from logging import LogRecord

# Centralises and Encapsulates all of the Aliasing and Configuration for the Logging System
class LoggerConfig:
    """
    Encapsulates configuration, enums, and type aliases for the logging system.
    This allows for easy collapsibility in IDEs and better code organization.
    """

    # CONSTANTS
    # Default Log File Directory
    LOG_DIR: Path = Path("/home/lloyd/EVIE/scripts/trading_bot/logs")
    # Default Log File Paths
    LOG_FILEPATHS: Dict[str, Path] = {
        "ASCII": LOG_DIR / "ASCII_log.txt",
        "coloured": LOG_DIR / "Coloured_ASCII_log.txt",
        "collapsible": LOG_DIR / "Collapsible_Coloured_ASCII_log.txt",
        "JSON": LOG_DIR / "Complex_JSON_log.json",
    }
    # Default Log File Names
    LOG_FILENAMES: Dict[str, str] = {
        "ASCII": "ASCII_log.txt",
        "coloured": "Coloured_ASCII_log.txt",
        "collapsible": "Collapsible_Coloured_ASCII_log.txt",
        "JSON": "Complex_JSON_log.json",
    }

    class LogLevel(Enum):
        NOTSET = "NOTSET"
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    # Correcting and refining type aliases
    Filename: TypeAlias = str
    Mode: TypeAlias = Literal["r", "w", "a", "r+", "w+", "a+", "x"]
    Color: TypeAlias = Tuple[int, int, int]
    LogEntryMessageType: TypeAlias = str
    LogEntryLevelType: TypeAlias = LogLevel  # Directly using LogLevel enum
    LogEntryModuleType: TypeAlias = Optional[str]
    LogEntryFunctionType: TypeAlias = Optional[str]
    LogEntryLineType: TypeAlias = Optional[int]
    LogEntryExceptionType: TypeAlias = BaseException
    LogEntryDictType: TypeAlias = Dict[
        str,
        Union[
            str,
            LogEntryLevelType,
            LogEntryModuleType,
            LogEntryFunctionType,
            LogEntryLineType,
            LogEntryExceptionType,
        ],
    ]
    LogEntryType: TypeAlias = LogRecord
    TimestampType: TypeAlias = str  # ISO 8601 format %Y-%m-%dT%H:%M:%S.%f

# Contains the methods for capturing enumeration information for logging to json, extensible to handle more types in the future.
class JsonLogTranscoder(json.JSONEncoder):
    """
    A custom JSON encoder that supports serialization of additional types.
    """

    def default(self, obj: Any) -> Any:
        """
        Serialize additional types to JSON.
        """
        # Custom serialization for Enum types
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        # Potential place for future custom serialization logic
        # Insert additional elif statements for other custom types here
        # Fallback to the base class implementation for other types
        return super().default(obj)

    @staticmethod
    def as_enum(dct: Dict[str, Any]) -> Any:
        """
        Deserialize JSON objects to Enum types.
        """
        if "__enum__" in dct:
            name, member = dct["__enum__"].split(".")
            return getattr(Enum, name)[member]
        return dct

# Contains the methods for capturing log information, default level debug, in a specified format for further procesing by the program. utilising the LoggingConfig class for all settings
class BasicLogger:
    """
    A basic logger class that captures log messages and splits them up into specific dictionary format including additional contextual metadata.
    """

    def __init__(
        self,
        timestamp: LoggerConfig.TimestampType = datetime.datetime.now().isoformat(),
        message: LoggerConfig.LogEntryMessageType = "",
        level: LoggerConfig.LogEntryLevelType = LoggerConfig.LogLevel.DEBUG,
        module: LoggerConfig.LogEntryModuleType = None,
        function: LoggerConfig.LogEntryFunctionType = None,
        line: LoggerConfig.LogEntryLineType = None,
        exc_info: LoggerConfig.LogEntryExceptionType = None,
    ) -> None:
        """
        Initializes a new instance of the BasicLogger class.
        Args:
            timestamp (TimestampType): The timestamp of the log entry. Defaults to the current timestamp.
            message (LogEntryMessageType): The log message to be captured.
            level (LogEntryLevelType): The log level of the message. Defaults to "DEBUG".
            module (LogEntryModuleType): The module where the log message originated. Defaults to None.
            function (LogEntryFunctionType): The function where the log message originated. Defaults to None.
            line (LogEntryLineType): The line number where the log message originated. Defaults to None.
            exc_info (LogEntryExceptionType): The exception information associated with the log message. Defaults to None.
        """
        self.timestamp: TimestampType = timestamp
        self.message: LogEntryMessageType = message
        self.level: LogEntryLevelType = level
        self.module: LogEntryModuleType = module
        self.function: LogEntryFunctionType = function
        self.line: LogEntryLineType = line
        self.exc_info: LogEntryExceptionType = exc_info

    def to_dict(self) -> LogEntryDictType:
        """
        Converts the LogEntry instance to a dictionary suitable for JSON serialization.
        Returns:
            JsonDict: The LogEntry instance as a dictionary.
        """
        return {
            "timestamp": self.timestamp,
            "message": self.message,
            "level": self.level,
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "exc_info": self.exc_info,
        }

    def to_json(self) -> str:
        """
        Converts the LogEntry instance to a JSON string.
        Returns:
            str: The LogEntry instance as a JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

# Takes the LogEntryDictType output from BasicLogger and returns a LogEntryDictType where each section of the LogEntryDict now contains fully formatted ASCII art graphics to enhance the formatting and presentation.
class AdvancedASCIIFormatter:
    """
    This Class takes log entries and automatically reformats them into nicely organised, bordered, boxed, seperated, delimited, embellished, organised entries with a clear header and footer and a clear seperator between each entry.
    The components making up the art of the log entry are the timestamp, level, message, module, function, line, and exception.
    The log entry is then formatted into a dictionary containing these components for further processing.
    The log entry is then formatted into a string with advanced ASCII formatting.
    This string is then returned as the output of the formatter, ready for colorisation and being sent to the terminal as well as conversion to a formatted, structured json object (each entry its own object so that the entries re collapsible) including the ASCII art elements.

    Takes: LogEntryDictType from BasicLogger and returns a LogEntryDictType where each section of the LogEntryDict now contains fully formatted ASCII art graphics to enhance the formatting and presentation.

    Attributes:
        message (LogEntryDictType): A dictionary containing the components of a log entry.
            - timestamp
            - level
            - message
            - module
            - function
            - line
            - exc_info
        ENTRY_SEPARATOR (str): A separator line to visually separate log entries.
        TOP_LEFT_CORNER (str): The top left corner character for the log entry box.
        TOP_RIGHT_CORNER (str): The top right corner character for the log entry box.
        BOTTOM_LEFT_CORNER (str): The bottom left corner character for the log entry box.
        BOTTOM_RIGHT_CORNER (str): The bottom right corner character for the log entry box.
        HORIZONTAL_LINE (str): The horizontal line character for the log entry box.
        VERTICAL_LINE (str): The vertical line character for the log entry box.
        HORIZONAL_DIVIDER_LEFT (str): The horizontal divider character for the left side of the log entry box.
        HORIZONAL_DIVIDER_RIGHT (str): The horizontal divider character for the right side of the log entry box.
        HORIZONTAL_DIVIDER_MIDDLE (str): The horizontal divider character for the middle of the log entry box.
        VERTICAL_DIVIDER (str): The vertical divider character for the log entry box.
        ERROR_INDICATOR_BOX_TOP_LEFT (str): The top left corner character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_BOTTOM_LEFT (str): The bottom left corner character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_TOP_RIGHT (str): The top right corner character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_BOTTOM_RIGHT (str): The bottom right corner character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_LEFT (str): The left side character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_RIGHT (str): The right side character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_MIDDLE (str): The middle character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_VERTICAL (str): The vertical character for the error indicator box inside the log entry box.
        ERROR_INDICATOR_BOX_HORIZONTAL (str): The horizontal character for the error indicator box inside the log entry box.
        LEVEL_SYMBOLS (Dict[LogLevel, str]): A mapping of log levels to their corresponding symbols.
        EXCEPTION_SYMBOLS (Dict[Type[Exception], str]): A mapping of exception types to their corresponding symbols.



    """

    def __init__(
        self,
        log: Optional[LoggerConfig.LogEntryDictType] = None,
    ) -> None:
        if log is None:
            log = {
                "timestamp": None,  # Placeholder for actual timestamp
                "message": None,  # Placeholder for actual message
                "level": None,  # Placeholder for actual level
                "module": None,  # Placeholder for actual module
                "function": None,  # Placeholder for actual function
                "line": None,  # Placeholder for actual line
                "exc_info": None,  # Placeholder for actual exception info
            }
        """
        Initializes the AdvancedASCIIFormatter class with the default ASCII art components.
        """
        self.log = log
        self.ENTRY_SEPARATOR: str = "-" * 80
        self.TOP_LEFT_CORNER: str = "┌"  # For Message Box Outline
        self.TOP_RIGHT_CORNER: str = "┐"  # For Message Box Outline
        self.BOTTOM_LEFT_CORNER: str = "└"  # For Message Box Outline
        self.BOTTOM_RIGHT_CORNER: str = "┘"  # For Message Box Outline
        self.HORIZONTAL_LINE: str = "─"  # For Message Box Outline
        self.VERTICAL_LINE: str = "│"  # For Message Box Outline
        self.HORIZONAL_DIVIDER_LEFT: str = (
            "├"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONAL_DIVIDER_RIGHT: str = (
            "┤"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONTAL_DIVIDER_MIDDLE: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.VERTICAL_DIVIDER: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_LEFT: str = (
            "╔"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_LEFT: str = (
            "╚"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_RIGHT: str = (
            "╗"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_RIGHT: str = (
            "╝"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_LEFT: str = (
            "╠"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_RIGHT: str = (
            "╣"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_MIDDLE: str = (
            "╬"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_VERTICAL: str = (
            "║"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_HORIZONTAL: str = (
            "═"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        # Level symbols for different log levels
        self.LEVEL_SYMBOLS: Dict[LogEntryLevelType, str] = {
            LoggerConfig.LogEntryLevelType.NOTSET: "🤷",
            LoggerConfig.LogEntryLevelType.DEBUG: "🐞",
            LoggerConfig.LogEntryLevelType.INFO: "ℹ️",
            LoggerConfig.LogEntryLevelType.WARNING: "⚠️",
            LoggerConfig.LogEntryLevelType.ERROR: "❌",
            LoggerConfig.LogEntryLevelType.CRITICAL: "🚨",
        }
        # Exception Symbols for different exception types
        self.EXCEPTION_SYMBOLS: Dict[Type[Exception], str] = {
            LoggerConfig.LogEntryExceptionType.ValueError: "#❌#",  # Value Error
            LoggerConfig.LogEntryExceptionType.TypeError: "T❌T",  # Type Error
            LoggerConfig.LogEntryExceptionType.KeyError: "K❌K",  # Key Error
            LoggerConfig.LogEntryExceptionType.IndexError: "I❌I",  # Index Error
            LoggerConfig.LogEntryExceptionType.AttributeError: "A❌A",  # Attribute Error
            LoggerConfig.LogEntryExceptionType.Exception: "E❌E",  # General Exception
        }

    def __call__(self, log_entry) -> dict:
        """
        Prepares a log entry dictionary with advanced ASCII formatting for further processing.

        Args:
            log_entry LogEntry: The log entry to process.

        Returns:
            dict: A dictionary containing the formatted log entry components for advanced ASCII formatting.

        Raises:
            TypeError: If the log entry is not an instance of LogEntry or CollapsibleLogEntry.
        """
        # Initialize a dictionary to hold the formatted log entry components
        ascii_log_entry_dict: dict = {}
        # Format a header for marking clearly the start of the log entry

        # Format the timestamp component of the log entry

        # Format the level component of the log entry

        # Format the message component of the log entry

        # Format the module component of the log entry

        # Format the function component of the log entry

        # Format the line component of the log entry

        # Format the exception component of the log entry

        # Add In the Authorship Details and a Footer Section to act as a clear seperator and identifier for the log entry end

        # Return the dictionary containing the formatted log entry components
        return ascii_log_entry_dict

    def format(self, log_entry) -> str:
        """
        This method runs through the entire formatting process for a message passed to it, ensuring that the message is recursively processed by each of the format methods as appropriate.
        Formats a log entry with advanced ASCII formatting. This method is a wrapper
        around the __call__ method to provide a 'format' interface.

        Args:
            log_entry (Union[LogEntry, CollapsibleLogEntry]): The log entry to format.

        Returns:
            str: The formatted log entry with advanced ASCII formatting.

        """
        return self.__call__(log_entry)
    

class JsonFormatter:
    """
    A formatter class that converts log entries to JSON format.

    This class inherits from the CollapsibleFormatter class and extends its functionality
    by converting the formatted log entries to a structured JSON format. The JSON format
    provides a more advanced and machine-readable representation of the log entries,
    enabling easier parsing and analysis of the logged data.

    The JsonFormatter class overrides the __call__ method to convert the log entry to
    a JSON string. It leverages the to_json() method of the log entry object to obtain
    the JSON representation of the log entry and its associated data.

    By utilizing the JsonFormatter, the logging system can produce structured and
    standardized log output in JSON format, facilitating integration with external
    tools and systems that rely on JSON-formatted data for further processing and analysis.

    Attributes:
        None

    Methods:
        __call__(self, log_entry: CollapsibleFormatter) -> str:
            Converts a log entry to a JSON string.

    Example Usage:
        formatter = JsonFormatter()
        json_log_entry = formatter(log_entry)
        print(json_log_entry)
    """

    def __call__(self, log_entry: CollapsibleFormatter = CollapsibleFormatter()) -> str:
        """
        Converts a log entry to a JSON string.

        This method overrides the __call__ method of the parent class to convert the
        formatted log entry to a JSON string. It utilizes the to_json() method of the
        log entry object to obtain the JSON representation of the log entry and its
        associated data.

        The JSON string representation of the log entry includes all the relevant
        information, such as the timestamp, log level, message, module, function,
        line number, and any additional metadata or context provided by the log entry.

        The resulting JSON string is returned as the output of the formatter, allowing
        it to be easily serialized, stored, or transmitted to external systems that
        expect JSON-formatted log data.

        Args:
            log_entry (CollapsibleFormatter): The log entry object to be converted to JSON.

        Returns:
            str: The log entry formatted as a JSON string.

        Raises:
            TypeError: If the provided log_entry is not an instance of CollapsibleFormatter.

        Example:
            log_entry = CollapsibleFormatter(...)
            json_formatter = JsonFormatter()
            json_log_entry = json_formatter(log_entry)
            print(json_log_entry)
        """
        # Check if the log entry is an instance of CollapsibleFormatter
        if not isinstance(log_entry, CollapsibleFormatter):
            raise TypeError(
                "JsonFormatter requires a CollapsibleFormatter log_entry object as input."
            )

        # Convert the log entry to a JSON string using the to_json() method
        json_log_entry: str = log_entry.to_json()

        # Return the JSON-formatted log entry
        return json_log_entry

class ColoredFormatter:
    """
    A formatter that adds color to the log output.
    """

    COLORS: Dict[LoggerConfig.LogLevel, Color] = {
        LoggerConfig.LogLevel.DEBUG: "\033[94m",
        LoggerConfig.LogLevel.INFO: "\033[92m",
        LoggerConfig.LogLevel.WARNING: "\033[93m",
        LoggerConfig.LogLevel.ERROR: "\033[91m",
        LoggerConfig.LogLevel.CRITICAL: "\033[95m",
    }
    RESET_COLOR: Color = "\033[0m"

    def __call__(
        self, log_entry: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """
        Formats a log entry with color based on its log level.

        Args:
            log_entry (Union[LogEntry, CollapsibleLogEntry]): The log entry to format.

        Returns:
            str: The formatted log entry with color.
        """
        color: Color = self.COLORS.get(log_entry.level, "")
        return super().__call__(
            f"{color}{log_entry.timestamp} | {log_entry.level.name.upper()} | {log_entry.message}{self.RESET_COLOR}"
        )


class DualOutput(Generic[T]):
    """

    """

    def __init__(
        self,
        
    ) -> None:
        """

        """
        
    def write() 
        """
        """

    def _format_console_message()
        """
        
        """
        

    def _format_file_message()
        """
        
        """
        

class DualLogger:
    """

    """

    def __init__(
     
    ) -> None:
        """
        
        """
        self.log_dir: Path = LOG_DIR
        self.stdout: TextIO = sys.stdout
        self.stderr: TextIO = sys.stderr

        # Default filter for maximum granularity

    def setup_logging(self) -> None:
        """
        Sets up the logging environment by creating necessary directories and files,
        and initializing log files for different formats. It also redirects stdout and stderr to the log files.
        """
        # Ensure the directory for each log file exists

        # Redirect stdout and stderr to the log files
        sys.stdout = self
        sys.stderr = self

        self.setup_non_blocking_logging()

    def setup_non_blocking_logging(self) -> None:
        """
        Sets up non-blocking logging using a queue mechanism. This method creates a logging queue,
        attaches a QueueHandler to the root logger, and starts a QueueListener to process log messages
        asynchronously using predefined handlers for console and file outputs.

        This setup ensures that logging operations do not block the main application flow, enhancing performance
        and responsiveness, especially in IO-bound or network-bound applications.
        """
        try:
            # Create a a rolling log queue with a maximum size of 1000 log entries

            # When 1000 entries have been hit in the queue (or if the queue is flushed for whatever reason) dump to file.

            # Maintain a maximum file size of 50mb and a maximum number of log files of 10

            # When File Size has been reached and file limit reached batch all 10 of the log files and compress them at maximum possible compression ratio and then vectorize the compressed data and store in a sqlite-vss database, ensuring the vectorization is reversible and similarity search can be used in the database. Generate embeddings from the vectors using an open source freely available embedding system available in python requiring no external access or apis.

            # Create a QueueHandler and attach it to the root logger

            # Define the BasicLogger file handler with BasicLogger

            # Define the ASCII file handler with the ASCII formatter

            # Define the JSON file handler with the JSON formatter

            # Define the Console Handlers for Coloured Formatted Output

            # Define the handlers that the QueueListener will use
            handlers = [

            ]

            # Create and start a QueueListener with respect_handler_level=True to respect each handler's log level
            queue_listener = QueueListener(
                log_queue, *handlers, respect_handler_level=True
            )
            queue_listener.start()

            # Store the listener in the class to ensure it can be stopped later
            self.queue_listener = queue_listener

        except Exception as e:
            # Log an error message or take appropriate action if setup fails
            logging.error(f"Failed to set up non-blocking logging: {e}")

    def write(self, message: str) -> None:
        """
        Writes a message to both the terminal(ASCII enhanced colourised) and the log file (each log file an object in json format ASCII enhanced).


        """
        
        

    def flush(self) -> None:
        """
        Flushes both the stdout and the log file.
        """
        self.stdout.flush()
        
    def log_message(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        

    def log_critical(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        Logs an informational message.

        Args:
            *args (Any): Positional arguments to be logged.
            **kwargs (Any): Keyword arguments to be logged.
        """
        self.log_message("=CRITICAL= ".joins(message), *args, **kwargs)

    def log_debug(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        self.log_message("=DEBUG= ".joins(message), *args, **kwargs)

    def log_info(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        self.log_message(("=INFO= " + " {message}"), *args, **kwargs)

    def log_warning(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        self.log_message(("=WARNING= " + " {message}"), *args, **kwargs)

    def log_error(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        self.log_message(("=ERROR= " + " {message}"), *args, **kwargs)

    def log_exception(self, message: str, *args: Any, **kwargs: Any) -> LogEntryDictType:
        """
        
        """
        self.log_error(message)
        traceback_str = "".join(
            traceback.format_exception(None, exception, exception.__traceback__)
        )
        self.log_error(traceback_str)

    def __enter__(self) -> "DualLogger":
        """
        Enters the runtime context related to this object.

        Returns:
            DualLogger: The DualLogger instance.
        """
        return self

    def __exit__(
        self,
        
    ) -> None:
        """
        Exits the runtime context related to this object.

        Args:
            exc_type (Optional[Type[BaseException]]): The type of the exception, if any.
            exc_value (Optional[BaseException]): The exception instance, if any.
            traceback (Optional[Any]): The traceback object, if any.
        """
        self.close()

    def close(self) -> None:
        """
        Closes the log file and restores the original stdout and stderr streams.
        """
        
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def main() -> None:
    """
    
    """
    # Set up the DualLogger with the default configuration


if __name__ == "__main__":
    # Call the main function when the script is run directly
    main()
