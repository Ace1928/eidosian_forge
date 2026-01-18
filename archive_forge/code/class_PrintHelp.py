from typing import final
class PrintHelp(Exception):
    """Raised when pytest should print its help to skip the rest of the
    argument parsing and validation."""