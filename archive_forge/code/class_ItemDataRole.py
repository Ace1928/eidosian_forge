import enum
import importlib
import_errors = []
class ItemDataRole(enum.Enum):
    EditRole = 1
    DisplayRole = 2
    ToolTipRole = 3
    ForegroundRole = 4