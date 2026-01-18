import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING

        Import assets from a module.

        args:
            path_parts: path parts to the assets directory (default: [])
            load_file: load the file (default: False)
            loader: loader function to use (default: None)
            **kwargs: additional arguments to pass to `get_module_assets`
        
        