import importlib
import importlib.metadata
import typing as t
import traceback
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _import_lib(self, name: str, package_name: t.Optional[str]=None) -> str:
    try:
        importlib.import_module(name)
        return importlib.metadata.version(package_name or name)
    except Exception:
        return traceback.format_exc()