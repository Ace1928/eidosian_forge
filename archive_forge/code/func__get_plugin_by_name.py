import importlib
import pkgutil
import os
import inspect
import asyncio
from typing import Dict, Type, Any, List, Optional, Union
from types import ModuleType, FunctionType
from abc import ABC, abstractmethod
from dependency_injector import containers, providers
import events  # Assumed robust asynchronous event handling.
import traceback
import json
import logging
from pydantic import BaseModel, create_model, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
def _get_plugin_by_name(self, plugin_name: str) -> Optional[AdvancedPluginInterface]:
    """
        Get a specific plugin by name, facilitating targeted management within the plugin ecosystem.
        """
    return self.plugins.get(plugin_name)